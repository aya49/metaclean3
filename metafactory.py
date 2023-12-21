# owner: alice yue
# created: 2022-09-05
# run on: Google Cloud Platform; metaflow-dev-sl-web project; ds-server.
# input: google cloud storage folders for .fcs and .csv files.
# output: load, clean (only metaclean others done in R), and cluster (framework only).

import os
import gcsfs  # google cloud storage access
import pickle
import copy

import pandas as pd
import numpy as np

import datetime
import time
import logging
import warnings

# import flowkit as fk  # loads .wsp files
import fcmdata_helpers as fdh

import fcsparser
# from MetaClean.quality_check import KernelClustersCleaner
from metalearner.dataprocessing._metaclean import BaysienDoubleKmeansCleaner

from utils import align_list, get_close_match, osrm

import matplotlib.pyplot as plt

import pdb


class Metafactory():
    """Load, clean, and save .fcs/.csv files from the Google Cloud Storage (GCS).

    Args:
        token_path (str): Google Cloud token path; for gcsfs.GCSFileSystem.
        project (:obj:`str`, optional): Google Cloud project that contains the buckets below.
        bucket_res (:obj:`str`, optional): Bucket to save results in.
        bucket_raw (:obj:`str`, optional): Bucket to load .fcs files from.
        bucket_csv (:obj:`str`, optional): Bucket to load .csv files (with labels) from.
        temp_local_dir (:obj:`str`, optional): Local directory in which temporary files are saved.

    Attributes:
        fs (:obj:`gcsfs.GCSFileSystem`): File system used to access GCS.
        bucket_res (:obj:`str`): Bucket to save results in.
        bucket_raw (:obj:`str`): Bucket to load .fcs files from.
        bucket_csv (:obj:`str`): Bucket to load .csv files (with labels) from.
        temp_local_dir (:obj:`str`): Local directory in which temporary files are saved.
        run_dir (:obj:`str`, optional): Directory in which final results are saved.
    """

    def __init__(
        self,
        token_path: str,  # '/home/maruko/projects/gss_key.json'
        project: str = 'metaflow-dev-sl-web',
        bucket_res: str = 'ds-metafora-results',
        bucket_raw: str = 'ds-datafactory-data-lake',
        bucket_csv: str = 'ds-datafactory-data-warehouse',
        temp_local_dir: str = '{}/temppy'.format(os.getcwd()),
    ):
        self.fs = gcsfs.GCSFileSystem(project=project, token=token_path)
        self.bucket_res = bucket_res
        self.bucket_raw = bucket_raw
        self.bucket_csv = bucket_csv
        self.temp_local_dir = temp_local_dir
        os.makedirs(self.temp_local_dir, exist_ok=True)


    def initialize_generator(
        self,
        data_set: str,  # 'Myeloma', 'Weber',
        file_type: str = 'csv'
    ):
        """Sets attributes data_set and data_set_folders listing folders to be processed.

        Args:
            data_set (str): See data_set attribute.
                You can also put the folder from the bucket_raw here
                if you do not need the cell population labels...
            file_type (str): type of file to load
        """

        self.data_set = data_set
        data_set_dir = '{}/{}/'.format(self.bucket_csv if file_type == 'csv'
            else self.bucket_raw, self.data_set)
        self.data_set_folders = self.fs.ls(data_set_dir.rstrip('/'))
        # sometimes needed, sometimes not
        self.data_set_folders = [dsf for dsf in self.data_set_folders
            if dsf != data_set_dir]

        return self


    def generate(
        self,
        run_dir: str = 'out',
        format_cp: bool = False,
        compensate: bool = True,
        markers: list = [],
        markers_rm: list = ['is_gate'],
        clean: str = '',
        settings_metaclean={
            'outlier_thresh': 0.01,
            'k_dtm': 50,
            'random_state': None,
            'remove_saturated': False,
            'thresh_vote': 0.01,
            'n_iter': 20,
        },
        label_col: str = 'label',
        get_label: bool = True,
        pkl_folder: str = '',
        overwrite: bool = True
    ):
        """Generate data set (generator function).

        Args:
            run_dir (str): Name of folder to save results in; used to output
                a path to where users can save their analysis into.
            format_cp (bool): See method do_load_gcp.
            compensate (bool): See method do_load.
            markers (list[str]): List of markers to keep.
            markers_rm (list[str]): List of markers to remove if exist.
            clean (str): see method do_clean_gcs.
            settings_metaclean (Dict): Settings for metaclean.
            label_col (:obj:`str`, optional): Column name for cell populations labels.
            overwrite (bool): Whether to overwrite files (upon load and clean).

        Returns:
            (tuple): Returns None if one step errors (e.g. no clean_index found).
                Otherwise, returns a tuple containing:
                - dat_clean (pandas.DataFrame): Compensated and cleaned (if specified) cell x `markers` matrix
                - meta (Dict): metadata from dat_clean's .fcs file.
                - cplabel_ (list[str]): Cell population label for each cell; may be str/int/float depending on data set.
                - run_path (str): Path to save user's analysis in; has no file extension.
        """
        self.get_label = get_label
        print('**************step1')
        run_directory = '{}/{}/{}{}{}'.format(
            self.bucket_res,
            self.data_set,
            run_dir,
            '_' if clean else '',
            clean)

        self.format_cp = format_cp
        self.compensate = compensate
        self.label_col = label_col
        self.overwrite = overwrite

        self.pkl_folder = pkl_folder #TODO

        self.clean = clean
        self.settings_metaclean = settings_metaclean
        print('**************step2')
        for self.data_set_folder in self.data_set_folders:
            data_set_filename = os.path.basename(self.data_set_folder)
            run_path = '{}/{}'.format(run_directory, data_set_filename)
            start = datetime.datetime.now()
            logging.info(data_set_filename)
            print('**************steps in for loop')
            # 01. load
            dat, meta, cplabel = self.do_load_gcs()
                # data_set_folder=data_set_folder,
                # fcs_folder,
                # temp_local_dir=self.temp_local_dir,
                # pkl_folder,
                # label_col: str = 'label',
                # compensate=compensate,
                # format_cp=format_cp,
                # overwrite=overwrite)
            logging.info('> cells: {}'.format(dat.shape[0]))
            logging.info('> load (s): {}'.format(
                (datetime.datetime.now() - start).total_seconds()))

            # clean markers
            # indicate column types (morpho, fluor, time),
            # keep only columns that end with 'a', because, just because
            # isolate specified markers

            # 02. clean
            dat_clean = dat.copy()
            cplabel_ = cplabel.copy() if cplabel is not None else None
            if clean:
                print('**************step in clean loop')
                print(dat.shape)
                print(meta)
                #to test
                if meta is None :
                   meta= {'$TIMESTEP': 0.001}

                clean_index = self.do_clean_gcs(
                    fcs=dat,
                    meta=meta,
                    # settings_metaclean=settings_metaclean,
                    # clean=clean,
                    # clean_file,
                    # png_dir_gcs,
                    # png_dir,
                    # overwrite=overwrite
                )

                if clean_index is None:
                    yield None
                    continue
                print('**************step 4')
                dat_clean = dat.iloc[clean_index]
                dat_clean.index = range(dat_clean.shape[0])

                cplabel_ = None
                if self.get_label:
                    if cplabel is not None:
                        cplabel_ = []
                        for ci in clean_index:
                            cplabel_.append(cplabel[ci])
                    else:
                        cplabel_ = None
                    # cplabel_ = [cplabel[ci] for ci in clean_index] if cplabel is not None else None

                logging.info('> clean (s): {}'.format(
                    (datetime.datetime.now() - start).total_seconds()))

            if len(markers_rm) > 0:
                print('**************step 5')
                for marker_rm in markers_rm:
                    if marker_rm in dat_clean.columns:
                        del(dat_clean[marker_rm])

            if len(markers) > 0:
                print('**************step6')
                markers_ = align_list(list1=list(dat.columns), list2=markers)
                time_chan = fdh.read.filter_channels(
                    dat, filter_type="time", return_data=False)

                dat_clean = dat_clean[markers_ + time_chan]

            yield dat_clean, meta, cplabel_, run_path


    def do_load_gcs(
        self,
        # data_set_folder: str,
        # temp_local_dir: str = '',
        # pkl_folder: str = '',
        # label_col: str = 'label',
        # compensate: bool = True,
        # format_cp: bool = False,
        # keep_csv_cols: bool,
        # overwrite: bool
    ):# -> tuple:
        """GCS wrapper for do_load(); saves results in GCS.

        Args:
            data_set_folder (str): See do_load().
            fcs_folder (:obj:`str`, optional): See do_load().
            temp_local_dir (:obj:`str`, optional): See do_load().
            pkl_folder (:obj:`str`, optional): GCS path to load/save data.
            label_col (:obj:`str`, optional): See do_load().
            format_cp (:obj:`bool`, optional): See do_load().
            compensate (:obj:`bool`, optional): See do_load().
            keep_csv_cols (:obj:`bool`, optional): See do_load().
            overwrite (:obj:`bool`, optional): Whether to overwrite file(s) if they exist already.

        Returns:
            (tuple): Same as do_load() except cplable is formatted if format_cp.
        """

        # set arguments
        os.makedirs(self.temp_local_dir, exist_ok=True)

        if not self.pkl_folder:
            self.pkl_folder = '{}/{}/data'.format(
                self.bucket_res,
                self.data_set)

        pkl_file = '{}/{}.pkl'.format(self.pkl_folder, os.path.basename(self.data_set_folder))
        print(pkl_file)
        # load and save data!
        if self.overwrite or not self.fs.exists(pkl_file.rstrip('/')):
            dat, meta, cplabel = self.do_load(get_label=self.get_label)
                # data_set_folder=data_set_folder,
                # temp_local_dir=temp_local_dir,
                # label_col=label_col,
                # compensate=compensate,
                # keep_csv_cols=keep_csv_cols)

            out = (dat, meta, cplabel)
            with self.fs.open(pkl_file, 'wb') as f:
                pickle.dump(out, f)

            cplabel_df = pd.DataFrame(cplabel, columns=['label'])
            label_file = '{}/{}/label/{}.csv'.format(
                self.bucket_res,
                self.data_set,
                os.path.basename(self.data_set_folder))
            with self.fs.open(label_file, 'wb', encoding='utf-8-sig') as f:
                cplabel_df.to_csv(f, index=False, header=False, encoding='utf-8-sig')
        else:
            with self.fs.open(pkl_file, 'rb') as f:
                dat, meta, cplabel = pickle.load(f)

        # format cell population labels e.g. lymphocytes/bcell -> bcell
        if self.format_cp and cplabel is not None and isinstance(cplabel[0], str):
            cplabel = [cp.split('/')[-1] for cp in cplabel]

        self.fs.clear_instance_cache()
        return dat, meta, cplabel

    def do_load(
        self,
        get_label=True,
        # data_set_folder: str,
        # temp_local_dir: str = '',
        # label_col: str = 'label',
        # compensate: bool = True,
        # keep_csv_cols: bool = True
    ):# -> tuple:
        """Load and compensate .csv and .fcs files from GCS; example path from Myeloma: gs://ds-datafactory-data-warehouse/Melyeloma/<patient_2-18,23-27>_Panel_<001-8>/PVTCOCHIN<000patient>.wsp | fcs_folder

        Args:
            data_set_folder (str): Path to directory containing .wsp file and
                fcs_folder/.csv to load (contains cell population labels).
                Also may be an element from attribute data_set_folders.
            temp_local_dir (:obj:`str`, optional): See attribute temp_local_dir.
            label_col (:obj:`str`, optional): Column name for cell populations labels.
            compensate (:obj:`bool`, optional): Whether or not to compensate.
                If True, returns compensated .fcs.
                IF False, returns loaded .csv without the label column.
            keep_csv_cols (:obj:`bool`, optional): Whether to keep .csv columns.

        Returns:
            (tuple): Returns tuple containing, for each flow sample:
                fcs (DataFrame[float]): cell x markers+time = compensated FI values.
                meta: Meta data from .fcs file.
                cplabel (list[str]): cell x label=cell/population/s.
        """

        # # set arguments
        # if not self.temp_local_dir:
        #     temp_local_dir = self.temp_local_dir

        ## load .csv file from fcs_folder
        fc_folder = [fc for fc in self.fs.ls(self.data_set_folder) if fc.endswith('_folder')]
        dat_paths = [dp for dp in self.fs.ls(fc_folder[0]) if dp[-1] != '/']

        # choose latest file only
        dat_path = dat_paths[0]
        if len(dat_paths) > 1:
            dat_path_date = self.fs.info(dat_path)['updated']
            for dat_path_ in dat_paths[1:]:
                dat_path_date_ = self.fs.info(dat_path_)['updated']
                if dat_path_date < dat_path_date_:
                    dat_path_date = dat_path_date_
                    dat_path = dat_path_

        cplabel = None
        raw = None

        raw_exist = False
        meta = None

        dat_path_ = dat_path
        if dat_path.endswith('.csv'):
            dat_path_ = dat_path_.replace('.csv', '.fcs').replace(self.bucket_csv, self.bucket_raw)

        fcs_exist = self.fs.exists(dat_path_)
        if (self.compensate or self.clean or not dat_path.endswith('.csv')) and self.fs.exists(dat_path_):
            # load .fcs file; metaclean requires meta['$TIMESTEP']; usually 1., in Myeloma, it's 0.01.
            fcs_file_local = '{}/{}'.format(self.temp_local_dir,
                os.path.basename(dat_path_))

            if fcs_exist:
                self.fs.get(dat_path_, fcs_file_local)
                meta, raw = fcsparser.parse(fcs_file_local, reformat_meta=False)
                osrm(fcs_file_local)

                # load labels
                if self.label_col in raw.columns:
                    cplabel = raw[self.label_col].to_list()

        # load .csv file
        dat_path_ = dat_path
        if not dat_path.endswith('.csv'):
            dat_path_ = dat_path_.replace('.fcs', '.csv').replace(self.bucket_raw, self.bucket_csv)
        if (dat_path.endswith('.csv') or (cplabel is None and get_label)) and self.fs.exists(dat_path_):
            raw_ = None
            try:
                raw_ = pd.read_csv(self.fs.open(dat_path_, 'r'), sep=',', header=0, nrows=5)
            except:
                print('.csv file is not delimited by ,')

            if raw_ is not None and isinstance(raw_, pd.DataFrame) and raw_.dtypes[0] != 'O' and not bool(set(raw_.columns)&set(raw_.iloc[0])):
                raw_ = pd.read_csv(self.fs.open(dat_path_, 'r'), sep=',', header=0)
            else:
                try:
                    raw_ = pd.read_csv(self.fs.open(dat_path_, 'r'), sep=';', header=0, nrows=5)
                except:
                    print('testing if .csv file is not delimited by , nor ;')

                if raw_ is not None and isinstance(raw_, pd.DataFrame) and raw_.dtypes[0] != 'O' and not bool(set(raw_.columns)&set(raw_.iloc[0])):
                    raw_ = pd.read_csv(self.fs.open(dat_path_, 'r'), sep=';', header=0)
                else:
                    raw_ = pd.read_csv(self.fs.open(dat_path_, 'r'), sep=';',
                                      header=0, skiprows=range(1, 2))

            if dat_path.endswith('.csv'):
                raw = raw_

            # load labels
            cplabel = None
            if get_label:
                cplabel = raw_[self.label_col].to_list()
            del (raw_[self.label_col])

        # ## 03. load .wsp and extract compensation matrix
        # wsp = fk.parse_wsp(fs.open('gs://' + ds_wf[0]))
        # fs.clear_instance_cache()

        # fgi = fcs_group_id if fcs_group_id != '' else list(wsp.items())[0][0]
        # pkl_folder_ = os.path.basename(dat_path).split('/')[-1]
        # pkl_folder = get_close_match(
        #     pkl_folder_, list(wsp.get(fgi).keys()))

        # spill = wsp.get(fgi).get(pkl_folder).get('compensation')
        # sd = align_list(list1=ldc, list2=spill.detectors)
        # sm = pd.DataFrame(spill.matrix, columns=sd)

        ## compensate = dat * comp^(-1)
        if self.compensate and fcs_exist:
            try:
                s = meta[get_close_match('spill', list(meta.keys()))].split(',')
                n = int(s[0])
                sd = align_list(list1=list(raw.columns), list2=s[1:(n+1)])  # just in case
                sm = np.array(s[(n+1):]).astype('float64').reshape(n, n)

                # correct diagonal of spill matrix
                smid = True
                for smi in range(sm.shape[0]):
                    if sm[smi,smi] != 1.0:
                        smid = False
                        break

                if smid:
                    sm += np.eye(sm.shape[0]) - np.diag(np.diag(sm))

                sm = pd.DataFrame(sm, columns=sd)

                # dat[sd] = np.dot(dat[sd], np.linalg.pinv(sm))
                raw[sd] = fdh.preprocessing.compensation.apply_compensation_matrix(raw[sd], sm)
            except:
                warnings.warn("File cannot be compensated.")

        if cplabel == None and self.get_label:
            label_path = '{}/{}/label/{}.csv'.format(
                self.bucket_res, self.data_set, os.path.basename(self.data_set_folder))
            if self.fs.exists(label_path) and self.fs.info(label_path)['size'] > 0:
                cplabel = pd.read_csv(self.fs.open(label_path, 'r'), header=None)[0].tolist()

        self.fs.clear_instance_cache()
        return raw, meta, cplabel


    def do_clean_gcs(
        self,
        fcs,
        meta = {'$TIMESTEP': 0.001},
        # clean: str = 'clean',
        # settings_metaclean = {
        #     'outlier_thresh': 0.01,
        #     'k_dtm': 50,
        #     'random_state': None,
        #     'remove_saturated': False,
        #     'thresh_vote': 0.01,
        #     'n_iter': 20,
        # },
        clean_file: str = '',
        png_dir_gcs: str = '',
        png_dir: str = '',
        # overwrite: bool = False
    ):
        """GCS wrapper for do_metaclean(); saves results in. GCS.

        Args:
            fcs (:obj:`pandas.DataFrame`): See method do_metaclean().
            meta (:obj:`Dictionary`, optional): See method do_metaclean().
            clean (:obj:`str`, optional): type of cleaning method to use; '' is none, 'clean' is metaclean, 'clean.peacoqc' is peacoqc.
            settings_clust (:obj:`Dictionary`, optional): See method do_metaclean().
            clean_file (:obj:`str`, optional): GCS path to load/save clean indices.
            png_dir_gcs (:obj:`str`, optional): GCS path to save .png plot.
            png_dir (:obj:`str`, optional): See method do_metaclean().
            overwrite (:obj:`bool`, optional): Whether to overwrite file(s) if they exist already.

        Returns:
            (:obj:`pandas.DataFrame`): See method do_clean().
        """

        # set arguments
        if not clean_file:
            clean_file = '{}/{}/{}/{}.csv'.format(
                self.bucket_res,
                self.data_set,
                self.clean,
                os.path.basename(self.data_set_folder))

        if not png_dir_gcs:
            png_dir_gcs = '{}/{}/{}_plot/{}.png'.format(
                self.bucket_res,
                self.data_set,
                self.clean,
                os.path.basename(self.data_set_folder))

        # if not png_dir:
        #     png_dir = '{}_png'.format(self.temp_local_dir)
        # os.makedirs(png_dir, exist_ok=True)

        # check if clean index exists
        if not self.fs.exists(clean_file) and (self.clean != 'clean' and self.clean != 'clean.20230619'):
            print(clean_file)
            warnings.warn('file not {}-ed yet, please first run 01_clean.R.'.format(self.clean))
            return None

        time_chan = fdh.read.filter_channels(
            fcs, filter_type="time", return_data=False)

        if (self.overwrite or not self.fs.exists(clean_file)) and self.clean == 'clean.20230619' and len(time_chan) > 0:
            print('  > start metaclean')
            start = time.time()
            clean_out = do_metaclean(
                fcs=fcs,
                meta=meta,
                settings_metaclean=self.settings_metaclean,
                png_dir=png_dir,
                # TODO: temporary work around for mass file in CleanPositiveControl
                true_channels=None if 'Mass' not in clean_file else np.array(['Marker_{}'.format(i) for i in range(1, 30)])
            )
            print('  > metaclean: {}\'s'.format(round(time.time() - start, 3)))

            if clean_out is None:
                warnings.warn('metaclean error')
                return None

            _, clean_index = clean_out
            clean_index = pd.DataFrame(np.where(np.array(clean_index))[0])

            with self.fs.open(clean_file, 'wb') as f:
                clean_index.to_csv(f, columns=None, header=False, index=False)
            if png_dir:
                self.fs.put(png_dir, png_dir_gcs)
                osrm(png_dir)

        # with self.fs.open(clean_file, 'r') as f:
        #     clean_index = pd.read_csv(f, header=None, index_col=None)[0]
        clean_index = fcs.index
        if len(time_chan) > 0:
            clean_index = pd.read_csv(self.fs.open(clean_file, 'r'), header=None, index_col=None)[0]
            clean_index = clean_index.astype(int).tolist()

        return clean_index


def do_metaclean(
    fcs,
    meta = {'$TIMESTEP': 1.0},
    settings_metaclean = {
        'outlier_thresh': 0.01,
        'k_dtm': 50,
        'random_state': None,
        'remove_saturated': False,
        'thresh_vote': 0.01,
        'n_iter': 20,
    },
    png_dir: str = '',
    true_channels = None,
    excluded_fluo_chans=['time','is_gate','label']
):
    """Copies (and deletes afterwards) .fcs to a local directory and performs metaclean on it.

    Taken from the metaclean main.

    Args:
        fcs (:obj:`pandas.DataFrame`): cell x markers+time = compensated FI values.
        meta (:obj:`Dictionary`, optional): the first element returned from fcsparser.parse(fcs_path).
        settings_clust (:obj:`Dictionary`, optional): Dictionary of MetaClean.quality_check.KernelClustersCleaner parameters (keys: 'kd' for 'k_dtm' and 'thresh_vote').
        png_dir (:obj:`str`, optional): Temporary file path to save clean plot; leave as '' if we don't want to plot.

    Returns:
        (tuple): Returns tuple containing:
            dat_clean (:obj:`pandas.DataFrame`): fcs but with low quality rows removed/cleaned.
            clean_index (:obj:`pandas.DataFrame`): A column with True for rows to keep and False for rows removed.
    """

    # set arguments
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)

    # metaclean over only fluo markers
    # if meta is None :
    #     fluo_channels = fdh.read.filter_channels(fcs, filter_type="fluo", return_data=False)
    # if len(fluo_channels) == 0:
    #     fluo_channels = [fm for fm in fcs.columns if 'Marker' in fm]
    fluo_channels= [col for col in fcs.columns if col.lower() not in excluded_fluo_chans
                    and not col.lower().startswith(('fsc', 'ssc'))]
    time_chan = fdh.read.filter_channels(
        fcs, filter_type="time", return_data=False)[0]

    logging.info('> all columns: {}'.format(', '.join(fcs.columns)))
    logging.info('> clean columns: {}'.format(', '.join(fluo_channels)))
    logging.info('> time column: {}'.format(time_chan))

    # metaclean
    # KCC = KernelClustersCleaner(
    #     fcs_data=fcs,
    #     metadata=meta,
    #     fluo_channels=fluo_channels,
    #     time_chan=time_chan,
    #     outlier_thresh=settings_metaclean['outlier_thresh'],
    #     k_dtm=settings_metaclean['k_dtm'],
    #     random_state=settings_metaclean['random_state'],
    #     remove_saturated=settings_metaclean['remove_saturated'])
    # orig_data, store_data = KCC.iteration_apply(iteration=3)
    # orig_data = KCC.preprocessing()
    # output is a pd.DataFrame that includes:
    #     index: ?
    #     index_orig: original index
    #     ORIGINAL DATA
    #     grp_flow: 4 markers > density > bin(which bin each cell was put into)
    #     weights: ?
    #     value: rescaled original data
    #     delete: outlier or not based on random forest
    #     attrib: keep, drop(includes outlier)
    # try:
    # dat_clean = KCC.ensemble_apply(
    #     keep="max_vote",
    #     thresh_vote=settings_metaclean['thresh_vote'],
    #     n_iter=settings_metaclean['n_iter'])
    # pdb.set_trace()
    # cleaner = BaysienDoubleKmeansCleaner(fluo_channels_man=fluo_channels)
    KCC = BaysienDoubleKmeansCleaner(k_dtm=settings_metaclean['k_dtm'], fluo_channels_man=fluo_channels,
                                     chan_to_keep=fluo_channels, remove_saturated=True,
                                     remove_zeros=False, config_remove='all')

    KCC.data = copy.copy(fcs)
    KCC.meta = copy.copy(meta)
    KCC.time_binned = KCC.get_flowconfig(true_channels)
    dat_clean_ = KCC.apply()
    # except:
    #     return None

    out = ()

    # plot
    # TODO(aya49) make more metaclean plots
    # if png_dir:
    #     clean_plot = plot_clean(dat_clean=dat_clean_, temp_png=png_dir)
    #     out = out + (clean_plot,)

    # format dat_clean
    dat_clean = pd.merge(KCC.data, dat_clean_, on='grp_flow', how='left')
    clean_index = dat_clean['keep']
    dat_clean = dat_clean[clean_index]
    dat_clean = dat_clean[fcs.columns]
    dat_clean.index = range(np.sum(clean_index))

    out = (dat_clean, clean_index) + out
    return out

#     fcs,
#     meta = {'$TIMESTEP': 1.0},
#     settings_metaclean = {
#         'outlier_thresh': 0.01,
#         'k_dtm': 50,
#         'random_state': None,
#         'remove_saturated': False,
#         'thresh_vote': 0.01,
#         'n_iter': 20,
#     },
#     png_dir: str = '',
#     true_channels = None

# def do_metaclean(fcs,
#                  meta= {'$TIMESTEP': 1.0},
#                 settings_metaclean={
#                     'outlier_thresh': 0.01,
#                     'k_dtm': 50,
#                     'random_state': None,
#                     'remove_saturated': False,
#                     'thresh_vote': 0.01,
#                     'n_iter': 20,
#                 },
#                 png_dir: str = '',
#                 true_channels = None
#                 ):
#     # metaclean over only fluo markers
#     fluo_channels = fdh.read.filter_channels(
#         fcs, filter_type="fluo", return_data=False)
#     if len(fluo_channels) == 0:
#         print('no fluo channel found')
#         fluo_channels = [fm for fm in fcs.columns if 'Marker' in fm]
#         print(fluo_channels)
#     logging.info('> all columns: {}'.format(', '.join(fcs.columns)))
#     logging.info('> clean columns: {}'.format(', '.join(fluo_channels)))
#     time_chan = fdh.read.filter_channels(
#         fcs, filter_type="time", return_data=False)[0]
# #     logging.info('> all columns: {}'.format(', '.join(fcs.columns)))
# #     logging.info('> clean columns: {}'.format(', '.join(fluo_channels)))
#     logging.info('MetaClean Starts')
#     start_time = time.time()

#     KCC = BaysienDoubleKmeansCleaner(k_dtm=settings_metaclean['k_dtm'], fluo_channels_man=fluo_channels,
#                                      remove_saturated=True, remove_zeros=False,
#                                      config_remove='all')

#     KCC.load_inputs(fcs, meta)
#     logging.info(
#         "--- METACLEAN {} seconds ---".format(time.time() - start_time))
#     print(KCC.data.shape)
#     # meta_index = KCC.data.loc[np.logical_not(
#     #     KCC.data['delete']), 'index_orig'].to_numpy()
#     # logging.info('MetaClean keep {} %'.format(
#     #     round(100*len(meta_index)/len(fcs), 2)))

#     # return np.sort(meta_index)
#     return KCC.data.index


def plot_clean(
    dat_clean: pd.DataFrame,
    temp_png: str
):
    """Plots metaclean output and saves it to local path.
    Taken from Nafise's AutoMyeloma.

    Args:
        dat_clean (pandas.DataFrame): _description_
        png_file (Str): Local path to save .png.

    Returns:
        (matplotlib.pyplot): metaclean plot
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 18))

    plot_data = dat_clean.loc[:, ['grp_flow', 'value', 'attrib']].groupby(
        'grp_flow').first()
    plot_data.reset_index(inplace=True)
    ratio_keep = sum((plot_data["attrib"] == "keep")) / plot_data.shape[0]
    title = "Keep: %.2f" % (ratio_keep)
    colors_map = {"keep": "darkblue", "drop": "red"}
    lg = []
    for labels, pts in plot_data.groupby("attrib")[["grp_flow", "value"]]:
        ax.scatter(pts["grp_flow"], pts["value"], c=colors_map[labels])
        lg.append(labels)
        ax.legend(lg)
        ax.title.set_text(title)
        ax.title.set_fontsize(15)
    plt.subplots_adjust(hspace=0.5)

    # # plotting script from metaclean function
    # dat_clean = dat_clean[["grp_flow", "value"]].drop_duplicates()
    # fig, ax = plt.subplots(figsize=(16, 8))
    # ax.scatter(dat_clean["grp_flow"], dat_clean["value"])

    # max_store = KCC.apply_regularized(keep='max_vote', iteration="auto", thresh=0.9)
    # fig, ax = plt.subplots(figsize=(16, 8))
    # ax.scatter(max_store["grp_flow"], max_store["value"])

    # orig_data, subdata = KCC.iteration_apply()
    # fig, ax = plt.subplots(figsize=(16, 8))
    # ax.scatter(subdata[0]["grp_flow"], subdata[0]["value"])

    os.makedirs(os.path.dirname(temp_png), exist_ok=True)
    fig.savefig(temp_png)

    return plot_data
