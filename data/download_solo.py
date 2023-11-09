import argparse
import logging
import os
import shutil
from datetime import timedelta, datetime
from multiprocessing import Pool
from urllib.request import urlopen
from warnings import simplefilter
from random import sample

import drms
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io.fits import getheader, HDUList
from dateutil.relativedelta import relativedelta
from sunpy.map import Map
from sunpy.net import Fido, attrs as a
import sunpy_soar
from tqdm import tqdm

class SOLODownloader:

    def __init__(self, base_path):
        self.base_path = base_path
        self.wavelengths = ['eui-hri174-image', 'eui-fsi174-image', 'eui-fsi304-image']
        self.dirs = ['eui-hri174-image', 'eui-fsi174-image', 'eui-fsi304-image']
        [os. makedirs(os.path.join(base_path, dir), exist_ok=True) for dir in self.dirs]

    def downloadDate(self, date):
        files = []
        try:
            # Download FSI
            for wl in self.wavelengths[1::]:
                files += [self.downloadFSI(date, wl)]
            #for wl in self.wavelengths[0]:
            #    files += [self.downloadHRI(date, wl)]
            logging.info('Download complete %s' % date.isoformat())
        except Exception as ex:
            logging.error('Unable to download %s: %s' % (date.isoformat(), str(ex)))
            [os.remove(f) for f in files if os.path.exists(f)]


    def downloadFSI(self, query_date, wl):
        file_path = os.path.join(self.base_path, wl, "%s.fits" % query_date.isoformat("T", timespec='seconds'))
        if os.path.exists(file_path):
            return file_path
        #
        search = Fido.search(a.Time(query_date - timedelta(minutes=15), query_date + timedelta(minutes=15)),
                             a.Instrument('EUI'), a.soar.Product(wl), a.Level(2))
        assert search.file_num > 0, "No data found for %s (%s)" % (query_date.isoformat(), wl)
        search = sorted(search['soar'], key=lambda x: abs(pd.to_datetime(x['Start time']) - query_date).total_seconds())
        #
        for entry in search:
            files = Fido.fetch(entry, path=self.base_path, progress=False)
            if len(files) != 1:
                continue
            file = files[0]

            # Clean data with header info or add printing meta data info
            header = getheader(file, 1)
            if header['CDELT1'] != 4.44012445:
                os.remove(file)
                continue

            shutil.move(file, file_path)
            return file_path

        raise Exception("No valid file found for %s (%s)!" % (query_date.isoformat(), wl))


    def downloadHRI(self, query_date, wl):
        file_path = os.path.join(self.base_path, wl, "%s.fits" % query_date.isoformat("T", timespec='seconds'))
        if os.path.exists(file_path):
            return file_path
        #
        search = Fido.search(a.Time(query_date + timedelta(minutes=15), query_date + timedelta(minutes=15)),
                             a.Instrument('EUI'), a.soar.Product(wl), a.Level(2))
        assert search.file_num > 0, "No data found for %s (%s)" % (query_date.isoformat(), wl)
        search = sorted(search['soar'], key=lambda x: abs(pd.to_datetime(x['Start Time']) - query_date).total_seconds())
        #
        for entry in search:
            files = Fido.fetch(entry, path=self.base_path, progress=False)
            if len(files) != 1:
                continue
            file = files[0]
            #header = Map(file.meta)


            shutil.move(file, file_path)
            return file_path

        raise Exception("No valid file found for %s (%s)!" % (query_date.isoformat(), wl))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Solar Orbiter data')
    parser.add_argument('--download_dir', type=str, help='path to the download directory.')
    parser.add_argument('--n_workers', type=str, help='number of parallel threads.', required=False, default=4)

    args = parser.parse_args()
    base_path = args.download_dir
    n_workers = args.n_workers
    #base_path = '/Users/christophschirninger/PycharmProjects/MDRAIT_ITI/'

    #[os.makedirs(os.path.join(base_path, str(c)), exist_ok=True) for c in ['hri_174', 'fsi_174', 'fsi_304']]
    download_util = SOLODownloader(base_path=base_path)
    start_date = datetime(2021, 2, 22, 0, 0)
    end_date = datetime(2021, 6, 25, 0, 0)
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    month_dates = [start_date + i * relativedelta(months=1) for i in range(num_months)]
    for date in month_dates:
        search = Fido.search(a.Time(date, date + relativedelta(days=1)),
                             a.Instrument('EUI'), a.soar.Product('eui-fsi174-image'),
                             a.Level(2))
        if search.file_num == 0:
            continue
        dates = search['soar']['Start time']
        dates = pd.to_datetime(dates)
        step = int(np.floor(len(dates) / 60)) if len(dates) > 60 else 1

        for d in dates[::step]:
            download_util.downloadDate(d)