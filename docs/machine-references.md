# Machine references and geometry provenance

Checked 20 November 2026. Use an exact model, region, blade and attachment configuration; dimensions from different generations are not interchangeable. Manufacturer PDFs are cached only under ignored `.local/references/`, not redistributed with this repository.

## Selected starting machine: Komatsu D61EX-24

The [official European Stage V D61EX/PX-24 brochure](https://www.komatsu.eu/Assets/GetBrochureByProductName.aspx?id=D61EX%2FPX-24&langID=en), printed page 17, supplies the dimensions in the EX and PX JSON profiles. The page's top/front/side drawings were visually checked. The source PDF SHA-256 at retrieval was `f1487805917db363b976439c76b2aa4f53598bb2aa6094771952e844c94b8297`.

The EX profile uses the standard 3.4 m³ INPAT blade: 3.250 m wide and 1.195 m high, on a 5.550 m overall machine. The PX profile separately uses the 3.8 m³ blade, 3.860 m wide and 1.155 m high. These are the cited European configuration, not a claim about a particular customer's machine.

Published dimensions and scenario assumptions are separated in each profile. The initial simulator uses blade width, face height and capacity; mass, track spacing and overall dimensions are metadata for later vehicle modeling. The effective fill factor and maximum push distance are uncalibrated simulation assumptions. Rated blade volume is not a measured per-pass carrying capacity.

## CAD and dimension drawings

The [official Komatsu Japan download directory](https://kcsj.komatsu/contact/download) lists D61EX-24 and D61PX-24 CAD:

- [D61EX-24 CAD request](https://go.kcsj.komatsu/l/682353/2022-03-13/7dh4t)
- [D61PX-24 CAD request](https://go.kcsj.komatsu/l/682353/2022-03-14/7dh9t)

Both links lead to a contact form. No personal information was submitted and no CAD file was obtained. The portal listing does not establish whether the downloadable asset is 2D or 3D, its format, or its reuse terms. Use the brochure's dimensioned views for coarse simulation geometry until the actual CAD and its terms can be reviewed. Do not infer cylinder pin locations, stroke ranges, center of mass, inertia, or detailed collision geometry from an exterior illustration.

## Cat D6N alternative

The [official Caterpillar-hosted D6N Specalog AEHQ7486-01](https://s7d2.scene7.com/is/content/Caterpillar/CM20150710-45266-06174) includes specifications and dimension drawings (printed pages 12-16). This is a useful alternative reference, not a CAD assembly. No verified manufacturer CAD download was obtained in this search. A D6N runtime profile is deferred until its complete selected blade geometry and configuration are verified; no values are borrowed from the D61.

The [official D6N Operation and Maintenance Manual preview SEBU7416-08](https://catpublications.com/previews/SEBU7416.pdf) identifies covered serial prefixes and publication date, but is only three opening pages. It is not the full operating, service or hydraulic manual. Obtain the complete serial-specific documentation through Caterpillar's publication/dealer channels before hardware integration.

## Komatsu manuals and future hydraulics

[My Komatsu](https://www.komatsu.com/en-us/services-and-support/equipment-monitoring-and-analysis/my-komatsu) provides machine-specific parts/service documentation. No authenticated full shop manual was obtained here. Access and serial applicability must be established before adopting service limits or hydraulic schematics.

The [2013 D61EXi/PXi-23 technical paper, section 2.2](https://www.komatsu.jp/en/-/media/home/aboutus/innovation/technology/techreport/2013/en/166_e02.pdf?hash=F8FBD4B5B23437C765E22216EF6F8277&rev=0dd9c6347d9c4780a5e4b6ac408be5fb) explains the cylinder-stroke/linkage/GNSS/IMU concept. It is a different machine generation and is not a source of D61-24 hydraulic tuning or linkage dimensions.

Current real-machine blade authority remains the user's external Trimble solution with two GNSS antennas. A future cylinder-feedback implementation requires verified linkage geometry, sensor calibration, actuator models, independent blade-edge measurements, and explicit control-authority management. The simulator sends no hardware commands.
