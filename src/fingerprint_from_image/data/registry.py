"""Registry of directly downloadable fingerprint datasets (no login / form / click-through).

Each :class:`RemoteFile` pins a URL, the SHA-256 we computed on download
(2026-10-22) and, where the host publishes one, its MD5.  :data:`DATASETS`
carries licence terms (with the URL they were read from), because research-
only / non-commercial terms propagate to models trained on the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["DATASETS", "DatasetSpec", "RemoteFile"]


@dataclass(frozen=True)
class RemoteFile:
    """One archive: where to get it and how to verify it."""

    url: str
    filename: str
    sha256: str | None
    size: int | None = None
    md5: str | None = None
    extract: bool = True  # False for documentation files


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset slug, its files and its terms."""

    slug: str
    title: str
    license: str  # short id written into every manifest row
    open_license: bool
    terms_url: str
    terms: str  # short quote / paraphrase of the governing terms
    files: tuple[RemoteFile, ...] = field(default_factory=tuple)
    downloader: str = "http"  # "http" or "gdrive" (public Google Drive file ids)


_ZENODO_UNSW = "https://zenodo.org/api/records/10432121/files/{}/content"
_GDRIVE = "https://drive.google.com/uc?id={}"

DATASETS: dict[str, DatasetSpec] = {
    spec.slug: spec
    for spec in (
        DatasetSpec(
            slug="fvc_full",
            title="FVC2000/2002/2004 DB1-4 sets A+B (Handbook of Fingerprint Recognition ESM)",
            license="FVC-research (no licence stated; (c) BioLab)",
            open_license=False,
            terms_url="http://bias.csr.unibo.it/fvc2004/download.asp",
            terms="Full databases are distributed with the Handbook (Springer 2021, ch. 4 ESM); "
                  "no licence text is given. Use for research, do not redistribute.",
            files=(RemoteFile(
                url="https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-030-83624-5_4"
                    "/MediaObjects/74034_3_En_4_MOESM1_ESM.zip",
                filename="fvc_handbook_esm.zip",
                sha256="075db0d32b5da7de872eb0714f6a873e04b1cba4f59d0aa678d7196e4611b4e5",
                size=869_928_818),),
        ),
        DatasetSpec(
            slug="unsw_2d3d",
            title="UNSW/UWA 2D and 3D Fingerprint Database (Zhou & Hu), contact + contactless",
            license="UNSW-2D3D research-only (Zenodo tag CC0, bundled licence stricter)",
            open_license=False,
            terms_url="https://zenodo.org/records/10432121",
            terms="'The use of this data/code is for the purpose of research only' ... cite "
                  "[1]-[4]; redistributions must reproduce the License agreement.",
            files=tuple(
                RemoteFile(url=_ZENODO_UNSW.format(name), filename=name, sha256=None,
                           size=size, md5=md5, extract=name.endswith(".7z"))
                for name, size, md5 in (
                    ("2D_Database.7z", 614_529_117, "39ce44a51b016e55904e9899918c11df"),
                    ("DS1.7z", 5_105_964_540, "2482be66b336d27e35636f0269c8abfb"),
                    ("DS2.7z", 5_486_367_223, "ecbb28527431406833ba3de3c2d0ba56"),
                    ("DS3.7z", 5_269_521_244, "bddc056ebf18d4ee036c065aea3febb6"),
                    ("README.md", 3014, "306f9efad07ab408e35c3823267a580e"),
                    ("2D_and_3D_Fingerprint_Database_License.pdf", 75421,
                     "70fffd265cb68a37246669d4781b817e"),
                )
            ),
        ),
        DatasetSpec(
            slug="impose_syn",
            title="IMPOSE-SYN: synthetic rolled gallery + multi-pose contactless queries",
            license="Apache-2.0",
            open_license=True,
            terms_url="https://github.com/Yu-Yy/IMPOSE",
            terms="Dataset README: 'License: Apache License 2.0.' (Pan et al., arXiv 2605.03830)",
            downloader="gdrive",
            files=(
                RemoteFile(url=_GDRIVE.format("1aB0R_pCLhWhBl0CXPsd-gv06LK9ZDR5f"),
                           filename="gallery.zip",
                           sha256="29f3bd38af923a8e3778869385c52651e805a3c2d050664fc5ce51b5928fb234",
                           size=260_195_610),
                RemoteFile(url=_GDRIVE.format("1uWkz3GQWGNbvtHUbchbTbU3-FpweGLEt"),
                           filename="query.zip",
                           sha256="d8b714bc0fcf010b3d50a655837d401ada0895c8b452063b38fb4f33eb363e5d",
                           size=2_097_839_868),
                RemoteFile(url=_GDRIVE.format("1C4V37JBrrut3LvypQJPvDh9zJT-S7vYX"),
                           filename="README.md",
                           sha256="786518a3c65a47d52fd4187682c9c2a156d11cf3fae1738c23d0eab52eb5869d",
                           size=973, extract=False),
            ),
        ),
        DatasetSpec(
            slug="l3sf",
            title="L3-SF level-3 synthetic fingerprints (Wyzykowski et al., ICPR 2020)",
            license="CC-BY-4.0",
            open_license=True,
            terms_url="https://zenodo.org/records/11098776",
            terms="Zenodo licence field: cc-by-4.0. Cite DOI 10.1109/ICPR48806.2021.9412304.",
            files=(RemoteFile(
                url="https://zenodo.org/api/records/11098776/files/L3_SF_V2.zip/content",
                filename="L3_SF_V2.zip",
                sha256="c66761b56317803b7051522e15575af64fa8fadfb8de9686c02b898a6832ca16",
                size=574_700_033, md5="1ae0d937ed882bfe61f2745bec7898f9"),),
        ),
        DatasetSpec(
            slug="tdfd",
            title="Tsinghua Distorted Fingerprint Database (320 normal/distorted pairs)",
            license="TDFD non-commercial research/education",
            open_license=False,
            terms_url="http://ivg.au.tsinghua.edu.cn/dataset/TDFD.php",
            terms="'You are free to download the dataset for non-commercial research and "
                  "educational purposes'; cite Si et al., TPAMI 2015.",
            files=(RemoteFile(
                url="http://ivg.au.tsinghua.edu.cn/dataset/samples_TDFD/TDFD/"
                    "Tsinghua%20Distorted%20Fingerprint%20Database.zip",
                filename="TDFD.zip",
                sha256="df75b0382fa124384a88603589f3e52b21e72e7620ced84bd721f1d4022fc8a3",
                size=75_698_565),),
        ),
        DatasetSpec(
            slug="neurotech",
            title="Neurotechnology sample fingerprint DBs (Cross Match Verifier 300, U.are.U 4000)",
            license="Neurotechnology sample (terms unstated)",
            open_license=False,
            terms_url="https://www.neurotechnology.com/download.html",
            terms="Public 'Sample fingerprint databases' downloads, 500 ppi; no terms stated.",
            files=(
                RemoteFile(url="https://www.neurotechnology.com/download/CrossMatch_Sample_DB.zip",
                           filename="CrossMatch_Sample_DB.zip",
                           sha256="aeb0a16e3a72db2daa7f73b65da3bf5509841d844c71949ecedf4b6cff5d88aa",
                           size=37_203_061),
                RemoteFile(url="https://www.neurotechnology.com/download/UareU_sample_DB.zip",
                           filename="UareU_sample_DB.zip",
                           sha256="fe141d8311853843405fabe6aed265e588366792a68913cad32dc92ba8f2043d",
                           size=59_671_809),
            ),
        ),
        DatasetSpec(
            slug="mendeley_family",
            title="FAMILY FINGERPRINT DATASET (Mendeley Data m4kz5ygxrn), 1 impression/finger",
            license="CC-BY-4.0",
            open_license=True,
            terms_url="https://data.mendeley.com/datasets/m4kz5ygxrn/1",
            terms="Mendeley data_licence: CC BY 4.0.",
            files=(RemoteFile(
                url="https://data.mendeley.com/public-files/datasets/m4kz5ygxrn/files/"
                    "d77aea16-3db8-4f8f-9f11-0e66ba84ccbb/file_downloaded",
                filename="family_fingerprint_dataset.zip",
                sha256="defefdd64aa2f403596792f1a1fbe880a29e5f853d12d39014f6ec86a4d329ea",
                size=223_775_430),),
        ),
        DatasetSpec(
            slug="naijafingset",
            title="NaijaFINGSET (Mendeley Data hzmn84khmh v2), 1,266 subjects, 1 impression",
            license="CC-BY-4.0",
            open_license=True,
            terms_url="https://data.mendeley.com/datasets/hzmn84khmh/2",
            terms="Mendeley data_licence: CC BY 4.0. SecuGen Hamster Plus scanner.",
            files=(RemoteFile(
                url="https://data.mendeley.com/public-api/zip/hzmn84khmh/download/2",
                filename="naijafingset_v2.zip",
                # Mendeley builds this zip on demand; if it is ever rebuilt the hash may
                # change while the 10,266 images stay the same.
                sha256="7d554f306e77904d5729993eccb3fb72749f93e076c5563cf9b752081314d278",
                size=257_369_729),),
        ),
    )
}
