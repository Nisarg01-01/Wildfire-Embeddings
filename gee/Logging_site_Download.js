/***************************************************************
 Logging_site_Download_v27_newsites.js
 
 PROTOCOL: unclipped_one (Solid 1km ROI, No Masking)
 DATA: HLS L30 v2 (0-1 Reflectance Scale)
 SITES: 12 verified sites.
 
 All pre/post dates set EXPLICITLY — no IDate-derived logic.
 This ensures pre is always before logging and post always after,
 regardless of any formula assumptions.
 
 Each site has:
   preStart  — start of 3-month pre-disturbance window (intact forest)
   postStart — start of 3-month post-disturbance window (logged area)
   Proof link in comment for independent verification.
***************************************************************/

var HLS_ID = 'NASA/HLS/HLSL30/v002';
var SCALE  = 30;
var EXPORT_FOLDER = 'GEE_Logging_v27_DeepForest_10';

var HLS = ee.ImageCollection(HLS_ID);

var sites = ee.List([

  // // ─────────────────────────────────────────────────────────────
  // 1. Toplita Valley, Romania — GPS from EuroNatur PDF
  // Proof: https://www.euronatur.org/fileadmin/docs/Urwald-Kampagne_
  //        Rumaenien/Photo_Documentation_Loggings_Romania_s_Nature_2000_Sites.pdf
  // PDF quote: "illegal forestry practices observed: logs transported through
  //   river beds, brutally built roads, very old trees marked for logging.
  //   Toplita valley." Photography: November 2018. Coords: 45.193541, 21.964540
  // Pre:  2017-06 → 2017-09 — intact old-growth valley forest
  // Post: 2019-06 → 2019-09 — post-logging landscape visible
  // ─────────────────────────────────────────────────────────────
  {id: 'log_2018_romania_toplita',
   lon: 21.964540, lat: 45.193541,
   preStart: '2018-03-01', postStart: '2019-03-01',
   name: 'Toplita Romania (PDF 45.193 21.964)'},

  // ─────────────────────────────────────────────────────────────
  // 2. Rondônia, Brazil — Amazon arc-of-deforestation
  // Proof: https://www.maapprogram.org/amazon-hotspots2021/
  //   (MAAP #153: Rondônia BR-364 corridor = primary 2021 hotspot)
  // Coords: -10.50°S, -62.85°W (BR-364 corridor, Rondônia state)
  // Dry season: May-September. Clearing ongoing 2021-2022.
  // Pre:  2020-06 → 2020-09 — intact Amazon primary forest, dry season
  // Post: 2022-06 → 2022-09 — cleared/degraded deforestation frontier
  // ─────────────────────────────────────────────────────────────
  {id: 'log_2021_rondonia',
   lon: -62.85, lat: -10.50,
   preStart: '2020-06-01', postStart: '2022-06-01',
   name: 'Rondonia Brazil'},

  // 3. Stung Tatai Leu Dam, Koh Kong, Cambodia
  // Proof: https://news.mongabay.com/2023/05/forest-behind-bars-logging-
  //        network-operating-out-of-cambodian-prison-in-the-cardamoms/
  //   (Mongabay May 2023: "Planet Labs/Earthrise satellite imagery from
  //   Feb 17 2023 shows clearing of forest for reservoir and road network.
  //   GFW: 116 deforestation alerts April 2022 → 6,000+ April 2023.")
  //        https://news.mongabay.com/2024/06/history-repeats-as-logging-
  //        linked-to-cambodian-hydropower-dam-in-cardamoms/
  //   (Jun 2024: "30 km of logging routes built Nov 2023-Jun 2024")
  // Coords: 11.25°N, 103.15°E (Stung Tatai Leu dam, Thma Bang, Koh Kong)
  // USER-SPECIFIED: Pre = Oct 2022 (intact, only 116 GFW alerts)
  //                 Post = Feb 2023 (satellite confirms active clearing)
  // Pre:  2022-10 → 2022-12 — intact Cardamom rainforest, dry season
  // Post: 2023-02 → 2023-04 — active clearing confirmed by satellite Feb 17
  // ─────────────────────────────────────────────────────────────
  {id: 'log_2022_cardamoms',
   lon: 103.38, lat: 11.75,
   preStart: '2022-02-01', postStart: '2023-03-15',
   name: 'Cardamoms Cambodia'}

]);

// ── PROCESSING ────────────────────────────────────────────────
// Uses explicit preStart/postStart from each site — no IDate formula.
var metaFC = ee.FeatureCollection(sites.map(function(site) {
  var s = ee.Dictionary(site);
  var pt = ee.Geometry.Point([s.get('lon'), s.get('lat')]);
  var roi = pt.buffer(1200).bounds();
  return ee.Feature(roi, s);
}));

// ── UI & EXPORT ───────────────────────────────────────────────
metaFC.evaluate(function(fc) {
  if (!fc) return;

  fc.features.forEach(function(f) {
    var p = f.properties;
    var roi = ee.Geometry(f.geometry);

    var pre  = HLS.filterBounds(roi)
                  .filterDate(p.preStart, ee.Date(p.preStart).advance(3, 'month'))
                  .median();
    var post = HLS.filterBounds(roi)
                  .filterDate(p.postStart, ee.Date(p.postStart).advance(3, 'month'))
                  .median();

    var dNDVI = pre.normalizedDifference(['B5','B4'])
                   .subtract(post.normalizedDifference(['B5','B4'])).rename('dNDVI');
    var dNBR  = pre.normalizedDifference(['B5','B7'])
                   .subtract(post.normalizedDifference(['B5','B7'])).rename('dNBR');
    var dNBR2 = pre.normalizedDifference(['B6','B7'])
                   .subtract(post.normalizedDifference(['B6','B7'])).rename('dNBR2');

    var visRGB = {bands: ['B4','B3','B2'], min: 0.02, max: 0.4, gamma: 1.3};
    var visIdx = {min: 0, max: 0.5, palette: ['white','yellow','red']};

    Map.addLayer(pre.clip(roi),   visRGB,   p.id + ' [PRE '  + p.preStart  + ']');
    Map.addLayer(post.clip(roi),  visRGB,   p.id + ' [POST ' + p.postStart + ']');
    Map.addLayer(dNDVI.clip(roi), visIdx,   p.id + ' dNDVI', false);
    Map.addLayer(dNBR.clip(roi),  visIdx,   p.id + ' dNBR',  false);
    Map.addLayer(dNBR2.clip(roi),
      {min: 0, max: 0.3, palette: ['blue','white','orange']},
      p.id + ' dNBR2 (Fire Guard)', false);

    Export.image.toDrive({
      image: pre.select(['B2','B3','B4','B5','B6','B7']).clip(roi).float(),
      description: p.id + '_PRE', folder: EXPORT_FOLDER, scale: 30, region: roi
    });
    Export.image.toDrive({
      image: post.select(['B2','B3','B4','B5','B6','B7']).clip(roi).float(),
      description: p.id + '_POST', folder: EXPORT_FOLDER, scale: 30, region: roi
    });
    Export.image.toDrive({
      image: dNDVI.addBands(dNBR).addBands(dNBR2).clip(roi).float(),
      description: p.id + '_INDICES_v27', folder: EXPORT_FOLDER, scale: 30, region: roi
    });
  });
});