"""Rebuild the CHLA-Z entries of datasets/data_collection.json from the
STAC items in datasets/chla_z/items/ (the maintained source).

Run after editing an item or adding a table: python src/update_data_collection.py
Other entries (IFCB) are kept as they are. Ids already published in
data_collection.json are kept; a new table needs a line in IDS.
"""
import copy, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GH = "https://github.com/fish-pace/fish-pace-datasets/blob/main/"
RAW = "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/"
path = REPO / "datasets/data_collection.json"
dc = json.loads(path.read_text())
coll = json.loads((REPO / "datasets/chla_z/collection.json").read_text())
item_files = [l["href"] for l in coll["links"] if l["rel"] == "item"]

# Keep the ids already published here; new tables get ids in the same style.
IDS = {
    "chla_argo_profiles": "global-bio-argo-chla-profile-metrics-0-200m-10m-bins",
    "chla_argo_profiles_plus_pace": "global-bio-argo-chla-profile-metrics-0-200m-10m-bins-with-pace-oci-matchups",
    "chla_ooi_profiles": "global-ooi-chla-profile-metrics-0-200m-10m-bins",
    "chla_ooi_profiles_plus_pace": "global-ooi-chla-profile-metrics-0-200m-10m-bins-with-pace-oci-matchups",
    "chla_argo_profiles_plus_olci": "global-bio-argo-chla-profile-metrics-0-200m-10m-bins-with-olci-matchups",
    "chla_ooi_profiles_plus_olci": "global-ooi-chla-profile-metrics-0-200m-10m-bins-with-olci-matchups",
}
LICENSE = {
    "argo": "Open access (Argo Data Policy); unrestricted use with attribution.",
    "ooi": "Open access; unrestricted use with attribution.",
}

chla = []
for rel in item_files:
    it = json.loads((REPO / "datasets/chla_z" / rel).read_text())
    p = copy.deepcopy(it["properties"])
    # The items' "How to load" text refers to the STAC browser's Copy URL
    # button; here the asset href is already absolute.
    p["description"] = p["description"].split("\n\nHow to load")[0]
    p["license"] = LICENSE["argo" if "argo" in Path(rel).stem else "ooi"]
    p["tutorial_notebook"] = GH + "datasets/chla_z/notebooks/" + p["tutorial_notebook"].split("/")[-1]
    a = it["assets"]["data"]
    chla.append({
        "type": "Feature", "stac_version": it["stac_version"], "id": IDS[Path(rel).stem],
        "properties": p, "geometry": it["geometry"], "bbox": it["bbox"],
        "assets": {"data": {"href": RAW + "datasets/chla_z/data/" + a["href"].split("/")[-1],
                            "type": a["type"], "roles": a["roles"], "title": p["title"]}},
        "links": [],
    })

others = [i for i in dc["items"] if "chla" not in i["id"]]
for i in others:  # IFCB: the notebook is Get_WHOI_IFCB.ipynb, not get_WHOI.ipynb
    nb = i["properties"].get("tutorial_notebook", "")
    if nb.endswith("/ifcb/notebooks/get_WHOI.ipynb"):
        i["properties"]["tutorial_notebook"] = nb.replace("get_WHOI.ipynb", "Get_WHOI_IFCB.ipynb")
dc["items"] = chla + others
dc["description"] = ("Datasets used in Fish-PACE tutorials and hackweeks: BGC-Argo and OOI "
                     "CHLA profiles with PACE and OLCI matchups (release 2026_11_23), and IFCB "
                     "daily abundance.")
dc["providers"][0]["url"] = "https://github.com/fish-pace/fish-pace-datasets"
path.write_text(json.dumps(dc, indent=2, ensure_ascii=False) + "\n")
print([i["id"] for i in dc["items"]])
