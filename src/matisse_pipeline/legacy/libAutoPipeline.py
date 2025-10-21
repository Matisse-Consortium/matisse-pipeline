# LEGACY FILE — DO NOT MODIFY
# This module is kept for backward compatibility with old data reduction workflows.
# Excluded from linters and pre-commit hooks.
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created in 2016
@author: pbe, fmillour, ame

Automatic MATISSE pipeline library !

Please contact florentin.millour@oca.eu for any question

This software is a computer program whose purpose is to show oifits
files from the MATISSE instrument.

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.

You can use, modify and/ or redistribute the software under the
terms of the CeCILL license as circulated by CEA, CNRS and INRIA at
the following URL "http://www.cecill.info". You have a copy of the
licence in the LICENCE.md file.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

import numpy as np
from astropy.io import fits
from astropy.io.fits import getheader
from astropy.time import Time


class headerCache:
    """"""

    # ----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.cache = {}
        self.max_cache_size = 1000

    # ----------------------------------------------------------------------
    def __contains__(self, key):
        """
        Returns True or False depending on whether or not the key is in the
        cache
        """
        return key in self.cache

    # ----------------------------------------------------------------------
    def update(self, key, value):
        self.cache[key] = {"value": value}

    # ----------------------------------------------------------------------
    @property
    def size(self):
        """
        Return the size of the cache
        """
        return len(self.cache)


cacheHdr = headerCache()


def matisseCalib(header, action, listCalibFile, calibPrevious, tplstart):
    global cacheHdr

    keyDetReadCurname = header["HIERARCH ESO DET READ CURNAME"]
    keyDetChipName = header["HIERARCH ESO DET CHIP NAME"]
    keyDetSeq1Dit = header["HIERARCH ESO DET SEQ1 DIT"]
    keyDetSeq1Period = header["HIERARCH ESO DET SEQ1 PERIOD"]
    keyInsPilId = header["HIERARCH ESO INS PIL ID"]
    keyInsPinId = header["HIERARCH ESO INS PIN ID"]
    keyInsDilId = header["HIERARCH ESO INS DIL ID"]
    keyInsDinId = header["HIERARCH ESO INS DIN ID"]
    keyInsPolId = header["HIERARCH ESO INS POL ID"]
    keyInsFilId = header["HIERARCH ESO INS FIL ID"]
    keyInsPonId = header["HIERARCH ESO INS PON ID"]
    keyInsFinId = header["HIERARCH ESO INS FIN ID"]
    keyDetMtrh2 = header["HIERARCH ESO DET WIN MTRH2"]
    keyDetMtrs2 = header["HIERARCH ESO DET WIN MTRS2"]
    res = calibPrevious
    if (
        action == "ACTION_MAT_CAL_DET_SLOW_SPEED"
        or action == "ACTION_MAT_CAL_DET_FAST_SPEED"
        or action == "ACTION_MAT_CAL_DET_LOW_GAIN"
        or action == "ACTION_MAT_CAL_DET_HIGH_GAIN"
    ):
        return [res, 1]

    allhdr = []
    for elt in listCalibFile:
        if elt not in cacheHdr:
            value = getheader(elt, 0)
            cacheHdr.update(elt, value)
        allhdr.append(cacheHdr.cache[elt]["value"])

    # Conversion of the tplstart of the correspoding recipe (mat_est_flat, mat_raw_estimates, ....) in mjd
    time_tplstart = Time(tplstart, format="isot", scale="utc")
    mjd_tplstart = time_tplstart.mjd

    if (
        action == "ACTION_MAT_IM_BASIC"
        or action == "ACTION_MAT_IM_EXTENDED"
        or action == "ACTION_MAT_IM_REM"
    ):
        nbCalib = 0
        for elt in res:
            if elt[1] == "BADPIX":
                nbCalib += 1

        for hdr, elt in zip(allhdr, listCalibFile, strict=False):
            tagCalib = matisseType(hdr)
            if tagCalib == "BADPIX":
                keyDetReadCurnameCalib = hdr["HIERARCH ESO DET READ CURNAME"]
                keyTplStartCalib = hdr["HIERARCH ESO TPL START"]
                keyDetChipNameCalib = hdr["HIERARCH ESO DET CHIP NAME"]
                time_tplstartcalib = Time(keyTplStartCalib, format="isot", scale="utc")
                mjd_tplstartcalib = time_tplstartcalib.mjd

            if tagCalib == "BADPIX" and (
                keyDetReadCurnameCalib == keyDetReadCurname
                and keyDetChipNameCalib == keyDetChipName
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1
        if nbCalib == 1:
            status = 1
        else:
            status = 0
        return [res, status]

    if action == "ACTION_MAT_EST_FLAT":
        nbCalib = 0
        for elt in res:
            if elt[1] == "BADPIX" or elt[1] == "FLATFIELD" or elt[1] == "NONLINEARITY":
                nbCalib += 1

        for hdr, elt in zip(allhdr, listCalibFile, strict=False):
            tagCalib = matisseType(hdr)
            if (
                tagCalib == "BADPIX"
                or tagCalib == "NONLINEARITY"
                or tagCalib == "FLATFIELD"
            ):
                keyDetReadCurnameCalib = hdr["HIERARCH ESO DET READ CURNAME"]
                keyDetChipNameCalib = hdr["HIERARCH ESO DET CHIP NAME"]
                keyDetSeq1DitCalib = hdr["HIERARCH ESO DET SEQ1 DIT"]
                keyTplStartCalib = hdr["HIERARCH ESO TPL START"]
                time_tplstartcalib = Time(keyTplStartCalib, format="isot", scale="utc")
                mjd_tplstartcalib = time_tplstartcalib.mjd

            if tagCalib == "BADPIX" and (
                keyDetReadCurnameCalib == keyDetReadCurname
                and keyDetChipNameCalib == keyDetChipName
            ):
                idx = -1
                cpt = 0

                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1

                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1
            if tagCalib == "FLATFIELD" and (
                (
                    keyDetChipNameCalib == "AQUARIUS"
                    and keyDetChipName == "AQUARIUS"
                    and keyDetReadCurnameCalib == keyDetReadCurname
                    and keyDetSeq1DitCalib == keyDetSeq1Dit
                )
                or (
                    keyDetChipNameCalib == "HAWAII-2RG"
                    and keyDetChipName == "HAWAII-2RG"
                    and keyDetReadCurnameCalib == keyDetReadCurname
                )
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1
            if tagCalib == "NONLINEARITY" and (
                (
                    keyDetChipNameCalib == "AQUARIUS"
                    and keyDetChipName == "AQUARIUS"
                    and keyDetReadCurnameCalib == keyDetReadCurname
                    and keyDetSeq1DitCalib == keyDetSeq1Dit
                )
                or (
                    keyDetChipNameCalib == "HAWAII-2RG"
                    and keyDetChipName == "HAWAII-2RG"
                    and keyDetReadCurnameCalib == keyDetReadCurname
                )
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1
        if nbCalib == 3:
            status = 1
        else:
            status = 0
        return [res, status]

    if action == "ACTION_MAT_RAW_ESTIMATES":
        nbCalib = 0
        for elt in res:
            if (
                elt[1] == "BADPIX"
                or elt[1] == "OBS_FLATFIELD"
                or elt[1] == "NONLINEARITY"
                or elt[1] == "SHIFT_MAP"
                or elt[1] == "KAPPA_MATRIX"
            ):
                nbCalib += 1

        for hdr, elt in zip(allhdr, listCalibFile, strict=False):
            tagCalib = matisseType(hdr)
            if (
                tagCalib == "BADPIX"
                or tagCalib == "OBS_FLATFIELD"
                or tagCalib == "NONLINEARITY"
                or tagCalib == "SHIFT_MAP"
                or tagCalib == "KAPPA_MATRIX"
            ):
                keyTplStartCalib = hdr["HIERARCH ESO TPL START"]
                keyDetReadCurnameCalib = hdr["HIERARCH ESO DET READ CURNAME"]
                keyDetChipNameCalib = hdr["HIERARCH ESO DET CHIP NAME"]
                keyDetSeq1DitCalib = hdr["HIERARCH ESO DET SEQ1 DIT"]
                keyInsPilIdCalib = hdr["HIERARCH ESO INS PIL ID"]
                keyInsPinIdCalib = hdr["HIERARCH ESO INS PIN ID"]
                keyInsDilIdCalib = hdr["HIERARCH ESO INS DIL ID"]
                keyInsDinIdCalib = hdr["HIERARCH ESO INS DIN ID"]
                keyInsPolIdCalib = hdr["HIERARCH ESO INS POL ID"]
                keyInsFilIdCalib = hdr["HIERARCH ESO INS FIL ID"]
                keyInsPonIdCalib = hdr["HIERARCH ESO INS PON ID"]
                keyInsFinIdCalib = hdr["HIERARCH ESO INS FIN ID"]
                keyDetMtrh2Calib = hdr["HIERARCH ESO DET WIN MTRH2"]
                keyDetMtrs2Calib = hdr["HIERARCH ESO DET WIN MTRS2"]
                time_tplstartcalib = Time(keyTplStartCalib, format="isot", scale="utc")
                mjd_tplstartcalib = time_tplstartcalib.mjd

            if tagCalib == "BADPIX" and (
                keyDetReadCurnameCalib == keyDetReadCurname
                and keyDetChipNameCalib == keyDetChipName
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1

            if tagCalib == "OBS_FLATFIELD" and (
                keyDetChipNameCalib == keyDetChipName
                and keyDetReadCurnameCalib == keyDetReadCurname
                and (
                    abs(keyDetSeq1DitCalib - keyDetSeq1Dit) < 0.001
                    or keyDetSeq1DitCalib == keyDetSeq1Period
                )
                and (
                    (
                        keyInsPilId == keyInsPilIdCalib
                        and keyInsDilId == keyInsDilIdCalib
                        and keyDetChipName == "HAWAII-2RG"
                        and keyDetReadCurname == "SCI-FAST-SPEED"
                    )
                    or (
                        keyInsPilId == keyInsPilIdCalib
                        and keyInsDilId == keyInsDilIdCalib
                        and keyDetChipName == "HAWAII-2RG"
                        and keyDetReadCurname == "SCI-SLOW-SPEED"
                        and keyDetMtrh2 == keyDetMtrh2Calib
                        and keyDetMtrs2 == keyDetMtrs2Calib
                    )
                    or (
                        keyInsPinId == keyInsPinIdCalib
                        and keyInsDinId == keyInsDinIdCalib
                        and keyDetChipName == "AQUARIUS"
                    )
                )
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1

            if tagCalib == "NONLINEARITY" and (
                (
                    keyDetChipNameCalib == "AQUARIUS"
                    and keyDetChipName == "AQUARIUS"
                    and keyDetReadCurnameCalib == keyDetReadCurname
                    and (
                        keyDetSeq1DitCalib == keyDetSeq1Dit
                        or keyDetSeq1DitCalib == keyDetSeq1Period
                    )
                )
                or (
                    keyDetChipNameCalib == "HAWAII-2RG"
                    and keyDetChipName == "HAWAII-2RG"
                    and keyDetReadCurnameCalib == keyDetReadCurname
                )
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1

            if tagCalib == "SHIFT_MAP" and (
                keyDetChipNameCalib == keyDetChipName
                and (
                    (
                        keyInsDilId == keyInsDilIdCalib
                        and keyInsFilId == keyInsFilIdCalib
                        and keyDetChipName == "HAWAII-2RG"
                        and keyInsDilId == "HIGH+"
                    )
                    or (
                        keyInsDilId == keyInsDilIdCalib
                        and keyDetChipName == "HAWAII-2RG"
                        and keyInsDilId != "HIGH+"
                    )
                    or (
                        keyInsDinId == keyInsDinIdCalib and keyDetChipName == "AQUARIUS"
                    )
                )
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1

            if tagCalib == "KAPPA_MATRIX" and (
                keyDetChipNameCalib == keyDetChipName
                and (
                    (
                        keyInsPolId == keyInsPolIdCalib
                        and keyInsDilId == keyInsDilIdCalib
                        and keyDetChipName == "HAWAII-2RG"
                    )
                    or
                    #                 ((keyInsPolId==keyInsPolIdCalib and keyInsFilId==keyInsFilIdCalib and keyInsDilId==keyInsDilIdCalib and keyDetChipName=="HAWAII-2RG") or
                    (
                        keyInsPonId == keyInsPonIdCalib
                        and keyInsFinId == keyInsFinIdCalib
                        and keyInsDinId == keyInsDinIdCalib
                        and keyDetChipName == "AQUARIUS"
                    )
                )
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1

            if tagCalib == "JSDC_CAT":
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx == -1:
                    res.append([elt, tagCalib])
        if (keyDetChipName == "AQUARIUS" and keyInsPinId == "PHOTO") or (
            keyDetChipName == "HAWAII-2RG" and keyInsPilId == "PHOTO"
        ):
            if nbCalib == 5:
                status = 1
            else:
                status = 0
        if (keyDetChipName == "AQUARIUS" and keyInsPinId != "PHOTO") or (
            keyDetChipName == "HAWAII-2RG" and keyInsPilId != "PHOTO"
        ):
            if nbCalib >= 4:
                status = 1
            else:
                status = 0
        return [res, status]

    if action == "ACTION_MAT_EST_KAPPA":
        nbCalib = 0
        for elt in res:
            if (
                elt[1] == "BADPIX"
                or elt[1] == "OBS_FLATFIELD"
                or elt[1] == "NONLINEARITY"
                or elt[1] == "SHIFT_MAP"
            ):
                nbCalib += 1
        for hdr, elt in zip(allhdr, listCalibFile, strict=False):
            tagCalib = matisseType(hdr)
            if (
                tagCalib == "BADPIX"
                or tagCalib == "NONLINEARITY"
                or tagCalib == "OBS_FLATFIELD"
                or tagCalib == "SHIFT_MAP"
            ):
                keyDetReadCurnameCalib = hdr["HIERARCH ESO DET READ CURNAME"]
                keyDetChipNameCalib = hdr["HIERARCH ESO DET CHIP NAME"]
                keyDetSeq1DitCalib = hdr["HIERARCH ESO DET SEQ1 DIT"]
                keyInsPilIdCalib = hdr["HIERARCH ESO INS PIL ID"]
                keyInsPinIdCalib = hdr["HIERARCH ESO INS PIN ID"]
                keyInsDilIdCalib = hdr["HIERARCH ESO INS DIL ID"]
                keyInsDinIdCalib = hdr["HIERARCH ESO INS DIN ID"]
                keyInsPolIdCalib = hdr["HIERARCH ESO INS POL ID"]
                keyInsFilIdCalib = hdr["HIERARCH ESO INS FIL ID"]
                keyInsPonIdCalib = hdr["HIERARCH ESO INS PON ID"]
                keyInsFinIdCalib = hdr["HIERARCH ESO INS FIN ID"]
                keyTplStartCalib = hdr["HIERARCH ESO TPL START"]
                time_tplstartcalib = Time(keyTplStartCalib, format="isot", scale="utc")
                mjd_tplstartcalib = time_tplstartcalib.mjd

            if tagCalib == "BADPIX" and (
                keyDetReadCurnameCalib == keyDetReadCurname
                and keyDetChipNameCalib == keyDetChipName
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1

            if tagCalib == "OBS_FLATFIELD" and (
                keyDetChipNameCalib == keyDetChipName
                and keyDetReadCurnameCalib == keyDetReadCurname
                and keyDetSeq1DitCalib == keyDetSeq1Dit
                and (
                    (
                        keyInsPilIdCalib == "PHOTO"
                        and keyInsDilId == keyInsDilIdCalib
                        and keyDetChipName == "HAWAII-2RG"
                    )
                    or (
                        keyInsPinIdCalib == "PHOTO"
                        and keyInsDinId == keyInsDinIdCalib
                        and keyDetChipName == "AQUARIUS"
                    )
                )
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1

            if tagCalib == "NONLINEARITY" and (
                (
                    keyDetChipNameCalib == "AQUARIUS"
                    and keyDetChipName == "AQUARIUS"
                    and keyDetReadCurnameCalib == keyDetReadCurname
                    and keyDetSeq1DitCalib == keyDetSeq1Dit
                )
                or (
                    keyDetChipNameCalib == "HAWAII-2RG"
                    and keyDetChipName == "HAWAII-2RG"
                    and keyDetReadCurnameCalib == keyDetReadCurname
                )
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1

            if tagCalib == "SHIFT_MAP" and (
                keyDetChipNameCalib == keyDetChipName
                and (
                    (
                        keyInsDilId == keyInsDilIdCalib
                        and keyDetChipName == "HAWAII-2RG"
                        and keyInsDilId != "HIGH+"
                    )
                    or (
                        keyInsDilId == keyInsDilIdCalib
                        and keyInsFilId == keyInsFilIdCalib
                        and keyDetChipName == "HAWAII-2RG"
                        and keyInsDilId == "HIGH+"
                    )
                    or (
                        keyInsDinId == keyInsDinIdCalib and keyDetChipName == "AQUARIUS"
                    )
                )
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    # print('tplstart = ',tplstart)
                    # print('tpl start previous=',keyTplStartPrevious,' with delta time = ',np.abs(mjd_tplstartprevious-mjd_tplstart))
                    # print('tpl start new=',keyTplStartCalib,' with delta time = ',np.abs(mjd_tplstartcalib-mjd_tplstart))
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                        # print('elt = ',elt)
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1
        if nbCalib == 4:
            status = 1
        else:
            status = 0
        return [res, status]

    if action == "ACTION_MAT_EST_SHIFT":
        nbCalib = 0
        for elt in res:
            if (
                elt[1] == "BADPIX"
                or elt[1] == "OBS_FLATFIELD"
                or elt[1] == "NONLINEARITY"
            ):
                nbCalib += 1
        for hdr, elt in zip(allhdr, listCalibFile, strict=False):
            tagCalib = matisseType(hdr)
            if (
                tagCalib == "BADPIX"
                or tagCalib == "NONLINEARITY"
                or tagCalib == "OBS_FLATFIELD"
            ):
                keyDetReadCurnameCalib = hdr["HIERARCH ESO DET READ CURNAME"]
                keyDetChipNameCalib = hdr["HIERARCH ESO DET CHIP NAME"]
                keyDetSeq1DitCalib = hdr["HIERARCH ESO DET SEQ1 DIT"]
                keyInsPilIdCalib = hdr["HIERARCH ESO INS PIL ID"]
                keyInsPinIdCalib = hdr["HIERARCH ESO INS PIN ID"]
                keyInsDilIdCalib = hdr["HIERARCH ESO INS DIL ID"]
                keyInsDinIdCalib = hdr["HIERARCH ESO INS DIN ID"]
                keyInsPolIdCalib = hdr["HIERARCH ESO INS POL ID"]
                keyInsFilIdCalib = hdr["HIERARCH ESO INS FIL ID"]
                keyInsPonIdCalib = hdr["HIERARCH ESO INS PON ID"]
                keyInsFinIdCalib = hdr["HIERARCH ESO INS FIN ID"]
                keyTplStartCalib = hdr["HIERARCH ESO TPL START"]
                time_tplstartcalib = Time(keyTplStartCalib, format="isot", scale="utc")
                mjd_tplstartcalib = time_tplstartcalib.mjd

            if tagCalib == "BADPIX" and (
                keyDetReadCurnameCalib == keyDetReadCurname
                and keyDetChipNameCalib == keyDetChipName
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1

            if tagCalib == "OBS_FLATFIELD" and (
                keyDetChipNameCalib == keyDetChipName
                and keyDetReadCurnameCalib == keyDetReadCurname
                and keyDetSeq1DitCalib == keyDetSeq1Dit
                and (
                    (
                        keyInsPilIdCalib == "PHOTO"
                        and keyInsDilId == keyInsDilIdCalib
                        and keyDetChipName == "HAWAII-2RG"
                    )
                    or (
                        keyInsPinIdCalib == "INTER"
                        and keyInsDinId == keyInsDinIdCalib
                        and keyDetChipName == "AQUARIUS"
                    )
                )
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1
            if tagCalib == "NONLINEARITY" and (
                (
                    keyDetChipNameCalib == "AQUARIUS"
                    and keyDetChipName == "AQUARIUS"
                    and keyDetReadCurnameCalib == keyDetReadCurname
                    and keyDetSeq1DitCalib == keyDetSeq1Dit
                )
                or (
                    keyDetChipNameCalib == "HAWAII-2RG"
                    and keyDetChipName == "HAWAII-2RG"
                    and keyDetReadCurnameCalib == keyDetReadCurname
                )
            ):
                idx = -1
                cpt = 0
                for elt2 in res:
                    if elt2[1] == tagCalib:
                        idx = cpt
                    cpt += 1
                if idx > -1:
                    hdu = fits.open(res[idx][0])
                    keyTplStartPrevious = hdu[0].header["HIERARCH ESO TPL START"]
                    time_tplstartprevious = Time(
                        keyTplStartPrevious, format="isot", scale="utc"
                    )
                    mjd_tplstartprevious = time_tplstartprevious.mjd
                    hdu.close()
                    if np.abs(mjd_tplstartcalib - mjd_tplstart) < np.abs(
                        mjd_tplstartprevious - mjd_tplstart
                    ):
                        del res[idx]
                        res.append([elt, tagCalib])
                else:
                    res.append([elt, tagCalib])
                    nbCalib += 1
        if nbCalib == 3:
            status = 1
        else:
            status = 0
        return [res, status]

    return [res, 0]


def matisseRecipes(action, det, tel, resol):
    if action == "ACTION_MAT_CAL_DET_SLOW_SPEED":
        return [
            "mat_cal_det",
            "--gain=2.73 --darklimit=100.0 --flatlimit=0.3 --max_nonlinear_range=36000.0 --max_abs_deviation=2000.0 --max_rel_deviation=0.01 --nltype=2",
        ]
    if action == "ACTION_MAT_CAL_DET_FAST_SPEED":
        return [
            "mat_cal_det",
            "--gain=2.60 --darklimit=100.0 --flatlimit=0.3 --max_nonlinear_range=36000.0 --max_abs_deviation=2000.0 --max_rel_deviation=0.01 --nltype=2",
        ]
    if action == "ACTION_MAT_CAL_DET_LOW_GAIN":
        return [
            "mat_cal_det",
            "--gain=190.0 --darklimit=100.0 --flatlimit=0.2 --max_nonlinear_range=36000.0 --max_abs_deviation=2000.0 --max_rel_deviation=0.02 --nt=true --nltype=2",
        ]
    if action == "ACTION_MAT_CAL_DET_HIGH_GAIN":
        return [
            "mat_cal_det",
            "--gain=20.0 --darklimit=200.0 --flatlimit=0.2 --max_nonlinear_range=36000.0 --max_abs_deviation=2000.0 --max_rel_deviation=0.01 --nt=true --nltype=2",
        ]
    if action == "ACTION_MAT_EST_FLAT":
        return ["mat_est_flat", "--obsflat_type=det"]
    if action == "ACTION_MAT_EST_SHIFT":
        return ["mat_est_shift", "--obsCorrection=TRUE"]
    if action == "ACTION_MAT_EST_KAPPA":
        return ["mat_est_kappa", ""]

    if action == "ACTION_MAT_RAW_ESTIMATES":
        options = ""

        if det == "AQUARIUS":
            options += "--useOpdMod=TRUE"
            if tel == "ESO-VLTI-A1234":
                options += " --replaceTel=3"

        elif det == "HAWAII-2RG":
            options += "--useOpdMod=FALSE --tartyp=57 --compensate=[pb,nl,if,rb,bp,od] --hampelFilterKernel=10"

        return ["mat_raw_estimates", options]

    if action == "ACTION_MAT_IM_BASIC":
        return ["mat_im_basic", ""]
    if action == "ACTION_MAT_IM_EXTENDED":
        return ["mat_im_extended", ""]
    if action == "ACTION_MAT_IM_REM":
        return ["mat_im_rem", ""]
    return ["", ""]


def matisseAction(header, tag):
    keyDetName = header["HIERARCH ESO DET NAME"]
    keyDetReadCurname = header["HIERARCH ESO DET READ CURNAME"]

    if (
        (tag == "DARK" or tag == "FLAT")
        and keyDetName == "MATISSE-LM"
        and keyDetReadCurname == "SCI-SLOW-SPEED"
    ):
        return "ACTION_MAT_CAL_DET_SLOW_SPEED"
    if (
        (tag == "DARK" or tag == "FLAT")
        and keyDetName == "MATISSE-LM"
        and keyDetReadCurname == "SCI-FAST-SPEED"
    ):
        return "ACTION_MAT_CAL_DET_FAST_SPEED"
    if (
        (tag == "DARK" or tag == "FLAT")
        and keyDetName == "MATISSE-N"
        and keyDetReadCurname == "SCI-LOW-GAIN"
    ):
        return "ACTION_MAT_CAL_DET_LOW_GAIN"
    if (
        (tag == "DARK" or tag == "FLAT")
        and keyDetName == "MATISSE-N"
        and keyDetReadCurname == "SCI-HIGH-GAIN"
    ):
        return "ACTION_MAT_CAL_DET_HIGH_GAIN"
    if tag == "OBSDARK" or tag == "OBSFLAT":
        return "ACTION_MAT_EST_FLAT"
    if (
        tag == "DISTOR_HOTDARK"
        or tag == "DISTOR_IMAGES"
        or tag == "SPECTRA_HOTDARK"
        or tag == "SPECTRA_IMAGES"
    ):
        return "ACTION_MAT_EST_SHIFT"
    if (
        tag == "KAPPA_HOTDARK"
        or tag == "KAPPA_SRC"
        or tag == "KAPPA_SKY"
        or tag == "KAPPA_OBJ"
    ):
        return "ACTION_MAT_EST_KAPPA"
    if (
        tag == "TARGET_RAW"
        or tag == "CALIB_RAW"
        or tag == "HOT_DARK"
        or tag == "CALIB_SRC_RAW"
        or tag == "SKY_RAW"
    ):
        return "ACTION_MAT_RAW_ESTIMATES"
    if tag == "IM_COLD":
        return "ACTION_MAT_IM_BASIC"
    if tag == "IM_FLAT" or tag == "IM_DARK":
        return "ACTION_MAT_IM_EXTENDED"
    if tag == "IM_PERIODIC":
        return "ACTION_MAT_IM_REM"
    return "NO-ACTION"


def matisseType(header):
    res = ""
    catg = None
    typ = None
    tech = None
    try:
        catg = header["HIERARCH ESO PRO CATG"]
    except:
        try:
            catg = header["HIERARCH ESO DPR CATG"]
            typ = header["HIERARCH ESO DPR TYPE"]
            tech = header["HIERARCH ESO DPR TECH"]
        except:
            pass
    if (catg == "CALIB" and typ == "DARK,DETCAL" and tech == "IMAGE") or (
        catg == "CALIB" and typ == "DARK" and tech == "IMAGE,DETCHAR"
    ):
        res = "DARK"
    elif (catg == "CALIB" and typ == "FLAT,DETCAL" and tech == "IMAGE") or (
        catg == "CALIB" and typ == "FLAT" and tech == "IMAGE,DETCHAR"
    ):
        res = "FLAT"
    # elif (catg=="CALIB" and (typ=="DARK" or typ=="FLAT,OFF") and tech=="SPECTRUM") :
    # Fix for when flat template is not run properly
    elif (
        (catg == "CALIB" or catg == "TEST")
        and (typ == "DARK" or typ == "FLAT,OFF")
        and tech == "SPECTRUM"
    ):
        res = "OBSDARK"
    elif (
        catg == "CALIB"
        and (typ == "FLAT" or typ == "FLAT,BLACKBODY")
        and tech == "SPECTRUM"
    ):
        res = "OBSFLAT"
    elif (catg == "CALIB" and typ == "DARK,WAVE" and tech == "IMAGE") or (
        catg == "CALIB" and typ == "DARK" and tech == "IMAGE"
    ):
        res = "DISTOR_HOTDARK"
    elif (catg == "CALIB" and typ == "SOURCE,WAVE" and tech == "IMAGE") or (
        catg == "CALIB" and typ == "WAVE,LAMP,PINHOLE" and tech == "SPECTRUM"
    ):
        res = "DISTOR_IMAGES"
    elif (catg == "CALIB" and typ == "SOURCE,LAMP" and tech == "SPECTRUM") or (
        catg == "CALIB" and typ == "WAVE,LAMP,SLIT" and tech == "SPECTRUM"
    ):
        res = "SPECTRA_HOTDARK"
    elif (catg == "CALIB" and typ == "SOURCE,WAVE" and tech == "SPECTRUM") or (
        catg == "CALIB" and typ == "WAVE,LAMP,FOIL" and tech == "SPECTRUM"
    ):
        res = "SPECTRA_IMAGES"
    elif (catg == "CALIB" and typ == "DARK,FLUX" and tech == "IMAGE") or (
        catg == "CALIB" and typ == "KAPPA,BACKGROUND" and tech == "SPECTRUM"
    ):
        res = "KAPPA_HOTDARK"
    elif (catg == "CALIB" and typ == "SOURCE,FLUX" and tech == "IMAGE") or (
        catg == "CALIB" and typ == "KAPPA,LAMP" and tech == "SPECTRUM"
    ):
        res = "KAPPA_SRC"
    elif catg == "SCIENCE" and typ == "OBJECT" and tech == "IMAGE":
        res = "TARGET_RAW"
    elif catg == "CALIB" and typ == "OBJECT" and tech == "IMAGE":
        res = "CALIB_RAW"
    elif (catg == "CALIB" and typ == "DARK,IMB" and tech == "IMAGE") or (
        catg == "CALIB" and typ == "DARK" and tech == "IMAGE,BASIC"
    ):
        res = "IM_COLD"
    elif (catg == "CALIB" and typ == "FLAT,IME" and tech == "IMAGE") or (
        catg == "CALIB" and typ == "FLAT" and tech == "IMAGE,EXTENDED"
    ):
        res = "IM_FLAT"
    elif (catg == "CALIB" and typ == "DARK,IME" and tech == "IMAGE") or (
        catg == "CALIB" and typ == "DARK" and tech == "IMAGE,EXTENDED"
    ):
        res = "IM_DARK"
    elif (catg == "CALIB" and typ == "DARK,FLAT" and tech == "IMAGE") or (
        catg == "CALIB" and typ == "FLAT,LAMP" and tech == "IMAGE,REMANENCE"
    ):
        res = "IM_PERIODIC"
    elif (
        catg == "CALIB"
        and (typ == "DARK" or typ == "BACKGROUND")
        and tech == "INTERFEROMETRY"
    ):
        res = "HOT_DARK"
    elif (
        catg == "CALIB"
        and (typ == "LAMP" or typ == "SOURCE" or typ == "SOURCE,FLUX")
        and tech == "INTERFEROMETRY"
    ):
        res = "CALIB_SRC_RAW"
    elif (
        (catg == "SCIENCE" or catg == "TEST")
        and typ == "OBJECT"
        and tech == "INTERFEROMETRY"
    ):
        res = "TARGET_RAW"
    elif (
        (catg == "TEST" and typ == "STD" and tech == "INTERFEROMETRY")
        or (catg == "CALIB" and typ == "OBJECT" and tech == "INTERFEROMETRY")
        or (catg == "CALIB" and typ == "OBJECT,FLUX" and tech == "INTERFEROMETRY")
        or (catg == "CALIB" and typ == "STD" and tech == "INTERFEROMETRY")
    ):
        res = "CALIB_RAW"
    elif (
        (catg == "TEST" or catg == "CALIB" or catg == "SCIENCE")
        and typ == "SKY"
        and tech == "INTERFEROMETRY"
    ):
        res = "SKY_RAW"
    else:
        res = catg
    return res
