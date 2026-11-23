"""OOI ERDDAP dataset ids used for CHLA-Z, moved out of ooi.ipynb cells 10 and 12 (ooi.ipynb is kept at tag 2026_11_23)
so the notebook and the build driver share one list.

OOI_DATASETS: dataset_id -> description. MOORING_CHL: ids whose CHLA variable is
`mass_concentration_of_chlorophyll_a_in_sea_water` (the rest use the
`..._profiler_depth_enabled` variable). OOI_SKIP_2025: the 2025 exclusion list,
kept for the record; `chla_z_build.OOI_EXCLUDE` is what the rebuild uses.
"""

OOI_DATASETS = {'ooi-ce04ossm-rid27-02-flortd000': 'Coastal Endurance: Oregon Offshore Surface Mooring: Near Surface Instrument '
                                    'Frame: 3-Wavelength Fluorometer',
 'ooi-ce04osps-sf01b-3a-flortd104': 'Coastal Endurance: Oregon Offshore Cabled Shallow Profiler Mooring: Shallow '
                                    'Profiler (SF01B): 3-Wavelength Fluorometer',
 'ooi-ce09ospm-wfp01-04-flortk000': 'Coastal Endurance: Washington Offshore Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-ce01issp-sp001-08-flortj000': 'Coastal Endurance: Oregon Inshore Surface Piercing Profiler Mooring: Surface '
                                    'Piercing Profiler: 3-Wavelength Fluorometer',
 'ooi-ce04ospd-dp01b-04-flntua103': 'Coastal Endurance: Oregon Offshore Cabled Deep Profiler Mooring: Wire-Following '
                                    'Profiler (DP01B): 2-Wavelength Fluorometer',
 'ooi-ce02shsp-sp001-07-flortj000': 'Coastal Endurance: Oregon Shelf Surface Piercing Profiler Mooring: Surface '
                                    'Piercing Profiler: 3-Wavelength Fluorometer',
 'ooi-ce02shsp-sp002-07-flortj000': 'Coastal Endurance: Oregon Shelf Surface Piercing Profiler Mooring: Surface '
                                    'Piercing Profiler: 3-Wavelength Fluorometer B',
 'ooi-ce06issp-sp001-08-flortj000': 'Coastal Endurance: Washington Inshore Surface Piercing Profiler Mooring: Surface '
                                    'Piercing Profiler: 3-Wavelength Fluorometer',
 'ooi-ce07shsp-sp001-07-flortj000': 'Coastal Endurance: Washington Shelf Surface Piercing Profiler Mooring: Surface '
                                    'Piercing Profiler: 3-Wavelength Fluorometer',
 'ooi-cp10cnsm-rid27-02-flortd000': 'Coastal Pioneer MAB: Central Surface Mooring: Near Surface Instrument Frame: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp13eapm-wfp01-04-flortk000': 'Coastal Pioneer MAB: Eastern Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp14nepm-wfp01-04-flortk000': 'Coastal Pioneer MAB: Northeastern Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp13nopm-wfp01-04-flortk000': 'Coastal Pioneer MAB: Northern Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp11nosm-rid27-02-flortd000': 'Coastal Pioneer MAB: Northern Surface Mooring: Near Surface Instrument Frame: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp14sepm-wfp01-04-flortk000': 'Coastal Pioneer MAB: Southeastern Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp13sopm-wfp01-04-flortk000': 'Coastal Pioneer MAB: Southern Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp11sosm-rid27-02-flortd000': 'Coastal Pioneer MAB: Southern Surface Mooring: Near Surface Instrument Frame: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp02pmci-wfp01-04-flortk000': 'Coastal Pioneer NES: Central Inshore Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp02pmco-wfp01-04-flortk000': 'Coastal Pioneer NES: Central Offshore Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp01cnpm-wfp01-04-flortk000': 'Coastal Pioneer NES: Central Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp01cnsm-rid27-02-flortd000': 'Coastal Pioneer NES: Central Surface Mooring: Near Surface Instrument Frame: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp01cnsp-sp001-09-flortj000': 'Coastal Pioneer NES: Central Surface Piercing Profiler Mooring: Surface Piercing '
                                    'Profiler: 3-Wavelength Fluorometer',
 'ooi-cp03ispm-wfp01-04-flortk000': 'Coastal Pioneer NES: Inshore Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp03issm-rid27-02-flortd000': 'Coastal Pioneer NES: Inshore Surface Mooring: Near Surface Instrument Frame: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp03issp-sp001-09-flortj000': 'Coastal Pioneer NES: Inshore Surface Piercing Profiler Mooring: Surface Piercing '
                                    'Profiler: 3-Wavelength Fluorometer',
 'ooi-cp04ospm-wfp01-04-flortk000': 'Coastal Pioneer NES: Offshore Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp04ossm-rid27-02-flortd000': 'Coastal Pioneer NES: Offshore Surface Mooring: Near Surface Instrument Frame: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp02pmui-wfp01-04-flortk000': 'Coastal Pioneer NES: Upstream Inshore Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-cp02pmuo-wfp01-04-flortk000': 'Coastal Pioneer NES: Upstream Offshore Profiler Mooring: Wire-Following Profiler: '
                                    '3-Wavelength Fluorometer',
 'ooi-rs03axps-pc03a-4c-flordd303': 'Regional Cabled Array: Axial Base Shallow Profiler Mooring: 200m Platform '
                                    '(PC03A): 2-Wavelength Fluorometer',
 'ooi-rs03axps-sf03a-3a-flortd301': 'Regional Cabled Array: Axial Base Shallow Profiler Mooring: Shallow Profiler '
                                    '(SF03A): 3-Wavelength Fluorometer',
 'ooi-rs01sbpd-dp01a-04-flntua102': 'Regional Cabled Array: Oregon Slope Base Deep Profiler Mooring: Wire-Following '
                                    'Profiler (DP01A): 2-Wavelength Fluorometer',
 'ooi-rs01sbps-pc01a-4c-flordd103': 'Regional Cabled Array: Oregon Slope Base Shallow Profiler Mooring: 200m Platform '
                                    '(PC01A): 2-Wavelength Fluorometer',
 'ooi-rs01sbps-sf01a-3a-flortd101': 'Regional Cabled Array: Oregon Slope Base Shallow Profiler Mooring: Shallow '
                                    'Profiler (SF01A): 3-Wavelength Fluorometer',
 'ooi-gp02hypm-wfp03-01-flordl000': 'Global Station Papa: Apex Profiler Mooring: Wire-Following Profiler Lower: '
                                    '2-Wavelength Fluorometer',
 'ooi-gp02hypm-wfp02-01-flordl000': 'Global Station Papa: Apex Profiler Mooring: Wire-Following Profiler Upper: '
                                    '2-Wavelength Fluorometer',
 'ooi-gp03flma-ris01-05-flortd000': 'Global Station Papa: Flanking Subsurface Mooring A: Mooring Riser: 3-Wavelength '
                                    'Fluorometer',
 'ooi-gp03flmb-ris01-05-flortd000': 'Global Station Papa: Flanking Subsurface Mooring B: Mooring Riser: 3-Wavelength '
                                    'Fluorometer',
 'ooi-gi02hypm-wfp02-01-flordl000': 'Global Irminger Sea: Apex Profiler Mooring: Wire-Following Profiler Upper: '
                                    '2-Wavelength Fluorometer',
 'ooi-gi01sumo-rii11-02-flordg033': 'Global Irminger Sea: Apex Surface Mooring: Mooring Riser: 2-Wavelength '
                                    'Fluorometer (130 meters)',
 'ooi-gi01sumo-rii11-02-flordg031': 'Global Irminger Sea: Apex Surface Mooring: Mooring Riser: 2-Wavelength '
                                    'Fluorometer (40 meters)',
 'ooi-gi01sumo-rii11-02-flordg032': 'Global Irminger Sea: Apex Surface Mooring: Mooring Riser: 2-Wavelength '
                                    'Fluorometer (80 meters)',
 'ooi-gi01sumo-rid16-02-flortd000': 'Global Irminger Sea: Apex Surface Mooring: Near Surface Instrument Frame: '
                                    '3-Wavelength Fluorometer',
 'ooi-gi01sumo-sbd12-02-flortd000': 'Global Irminger Sea: Apex Surface Mooring: Surface Buoy: 3-Wavelength Fluorometer',
 'ooi-gi03flma-ris01-05-flortd000': 'Global Irminger Sea: Flanking Subsurface Mooring A: Mooring Riser: 3-Wavelength '
                                    'Fluorometer',
 'ooi-gi03flmb-ris01-05-flortd000': 'Global Irminger Sea: Flanking Subsurface Mooring B: Mooring Riser: 3-Wavelength '
                                    'Fluorometer',
 'ooi-ga02hypm-wfp03-01-flordl000': 'Global Argentine Basin: Apex Profiler Mooring: Wire-Following Profiler Lower: '
                                    '2-Wavelength Fluorometer',
 'ooi-ga02hypm-wfp02-01-flordl000': 'Global Argentine Basin: Apex Profiler Mooring: Wire-Following Profiler Upper: '
                                    '2-Wavelength Fluorometer',
 'ooi-ga01sumo-rii11-02-flordg033': 'Global Argentine Basin: Apex Surface Mooring: Mooring Riser: 2-Wavelength '
                                    'Fluorometer (130 meters)',
 'ooi-ga01sumo-rii11-02-flordg031': 'Global Argentine Basin: Apex Surface Mooring: Mooring Riser: 2-Wavelength '
                                    'Fluorometer (40 meters)',
 'ooi-ga01sumo-rii11-02-flordg032': 'Global Argentine Basin: Apex Surface Mooring: Mooring Riser: 2-Wavelength '
                                    'Fluorometer (80 meters)',
 'ooi-ga01sumo-rid16-02-flortd000': 'Global Argentine Basin: Apex Surface Mooring: Near Surface Instrument Frame: '
                                    '3-Wavelength Fluorometer',
 'ooi-ga01sumo-sbd12-02-flortd000': 'Global Argentine Basin: Apex Surface Mooring: Surface Buoy: 3-Wavelength '
                                    'Fluorometer',
 'ooi-ga03flma-ris01-05-flortd000': 'Global Argentine Basin: Flanking Subsurface Mooring A: Mooring Riser: '
                                    '3-Wavelength Fluorometer',
 'ooi-ga03flmb-ris01-05-flortd000': 'Global Argentine Basin: Flanking Subsurface Mooring B: Mooring Riser: '
                                    '3-Wavelength Fluorometer',
 'ooi-gs02hypm-wfp03-01-flordl000': 'Global Southern Ocean: Apex Profiler Mooring: Wire-Following Profiler Lower: '
                                    '2-Wavelength Fluorometer',
 'ooi-gs02hypm-wfp02-01-flordl000': 'Global Southern Ocean: Apex Profiler Mooring: Wire-Following Profiler Upper: '
                                    '2-Wavelength Fluorometer',
 'ooi-gs01sumo-rii11-02-flordg033': 'Global Southern Ocean: Apex Surface Mooring: Mooring Riser: 2-Wavelength '
                                    'Fluorometer (130 meters)',
 'ooi-gs01sumo-rii11-02-flordg031': 'Global Southern Ocean: Apex Surface Mooring: Mooring Riser: 2-Wavelength '
                                    'Fluorometer (40 meters)',
 'ooi-gs01sumo-rii11-02-flordg032': 'Global Southern Ocean: Apex Surface Mooring: Mooring Riser: 2-Wavelength '
                                    'Fluorometer (80 meters)',
 'ooi-gs01sumo-rid16-02-flortd000': 'Global Southern Ocean: Apex Surface Mooring: Near Surface Instrument Frame: '
                                    '3-Wavelength Fluorometer',
 'ooi-gs01sumo-sbd12-02-flortd000': 'Global Southern Ocean: Apex Surface Mooring: Surface Buoy: 3-Wavelength '
                                    'Fluorometer',
 'ooi-gs03flma-ris01-05-flortd000': 'Global Southern Ocean: Flanking Subsurface Mooring A: Mooring Riser: 3-Wavelength '
                                    'Fluorometer',
 'ooi-gs03flmb-ris01-05-flortd000': 'Global Southern Ocean: Flanking Subsurface Mooring B: Mooring Riser: 3-Wavelength '
                                    'Fluorometer'}

MOORING_CHL = ['ooi-ce04ossm-rid27-02-flortd000',
 'ooi-cp10cnsm-rid27-02-flortd000',
 'ooi-cp11nosm-rid27-02-flortd000',
 'ooi-cp11sosm-rid27-02-flortd000',
 'ooi-cp01cnsm-rid27-02-flortd000',
 'ooi-cp03issm-rid27-02-flortd000',
 'ooi-cp04ossm-rid27-02-flortd000',
 'ooi-rs03axps-pc03a-4c-flordd303',
 'ooi-rs01sbps-pc01a-4c-flordd103',
 'ooi-gp03flma-ris01-05-flortd000',
 'ooi-gp03flmb-ris01-05-flortd000',
 'ooi-gi01sumo-rii11-02-flordg033',
 'ooi-gi01sumo-rii11-02-flordg031',
 'ooi-gi01sumo-rii11-02-flordg032',
 'ooi-gi01sumo-rid16-02-flortd000',
 'ooi-gi01sumo-sbd12-02-flortd000',
 'ooi-gi03flma-ris01-05-flortd000',
 'ooi-gi03flmb-ris01-05-flortd000',
 'ooi-ga01sumo-rii11-02-flordg033',
 'ooi-ga01sumo-rii11-02-flordg031',
 'ooi-ga01sumo-rii11-02-flordg032',
 'ooi-ga01sumo-rid16-02-flortd000',
 'ooi-ga01sumo-sbd12-02-flortd000',
 'ooi-ga03flma-ris01-05-flortd000',
 'ooi-ga03flmb-ris01-05-flortd000',
 'ooi-gs01sumo-rii11-02-flordg033',
 'ooi-gs01sumo-rii11-02-flordg031',
 'ooi-gs01sumo-rii11-02-flordg032',
 'ooi-gs01sumo-rid16-02-flortd000',
 'ooi-gs01sumo-sbd12-02-flortd000',
 'ooi-gs03flma-ris01-05-flortd000',
 'ooi-gs03flmb-ris01-05-flortd000']

OOI_SKIP_2025 = ['ooi-cp02pmci-wfp01-04-flortk000',
 'ooi-cp02pmco-wfp01-04-flortk000',
 'ooi-cp01cnpm-wfp01-04-flortk000',
 'ooi-cp01cnsm-rid27-02-flortd000',
 'ooi-cp01cnsp-sp001-09-flortj000',
 'ooi-cp03ispm-wfp01-04-flortk000',
 'ooi-cp03issm-rid27-02-flortd000',
 'ooi-cp03issp-sp001-09-flortj000',
 'ooi-cp04ospm-wfp01-04-flortk000',
 'ooi-cp04ossm-rid27-02-flortd000',
 'ooi-cp02pmui-wfp01-04-flortk000',
 'ooi-cp02pmuo-wfp01-04-flortk000',
 'ooi-rs01sbpd-dp01a-04-flntua102',
 'ooi-ga02hypm-wfp03-01-flordl000',
 'ooi-ga02hypm-wfp02-01-flordl000',
 'ooi-ga01sumo-rii11-02-flordg033',
 'ooi-ga01sumo-rii11-02-flordg031',
 'ooi-ga01sumo-rii11-02-flordg032',
 'ooi-ga01sumo-rid16-02-flortd000',
 'ooi-ga01sumo-sbd12-02-flortd000',
 'ooi-ga03flma-ris01-05-flortd000',
 'ooi-ga03flmb-ris01-05-flortd000',
 'ooi-gs02hypm-wfp03-01-flordl000',
 'ooi-gs02hypm-wfp02-01-flordl000',
 'ooi-gs01sumo-rii11-02-flordg033',
 'ooi-gs01sumo-rii11-02-flordg031',
 'ooi-gs01sumo-rii11-02-flordg032',
 'ooi-gs01sumo-rid16-02-flortd000',
 'ooi-gs01sumo-sbd12-02-flortd000',
 'ooi-gs03flma-ris01-05-flortd000',
 'ooi-gs03flmb-ris01-05-flortd000',
 'ooi-ce02shsp-sp002-07-flortj000',
 'ooi-rs01sbps-pc01a-4c-flordd103',
 'ooi-gp02hypm-wfp03-01-flordl000']
