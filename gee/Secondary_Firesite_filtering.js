// Load fire list and set quality thresholds for satellite validation
var incidents = ee.FeatureCollection("users/nshah132/Global_fire_incidents_Int");
var COLLECTION_ID = "NASA/HLS/HLSL30/v002";  // Harmonized Landsat Sentinel-2: 30m resolution, 2-3 day revisit
var TARGET_COUNT = 50;  // Final output: 50 high-quality fire sites
var CLOUD_THRESHOLD = 10;  // Filter: only accept scenes with <10% cloud cover (used in: filter(ee.Filter.lt('CLOUD_COVERAGE', CLOUD_THRESHOLD)))
                           
var MIN_MODERATE_BURN_FRACTION = 0.4;  // Filter: require ≥40% of ROI with dNBR > 0.4 (used in: safeFraction.gte(MIN_MODERATE_BURN_FRACTION))
                                        
var MIN_VALID_COVERAGE = 0.9;  // Filter: require 90% valid pixels in BOTH pre and post images (used in: preCoverage.gte(MIN_VALID_COVERAGE) && postCoverage.gte(MIN_VALID_COVERAGE))

var rgbVis = {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3, gamma: 1.2};
var dnbrVis = {min: -0.1, max: 0.7, palette: ['green', 'yellow', 'orange', 'red']};

// Create water mask to exclude pixels over water
var waterMask = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
                  .select('occurrence')
                  .unmask(0)
                  .gte(50)
                  .not();

// Clean satellite images: remove clouds, shadows, and snow
var maskHLS = function(image) {
  var qa = image.select('Fmask'); 
  var mask = qa.bitwiseAnd(1 << 1).eq(0)  // Cloud
    .and(qa.bitwiseAnd(1 << 3).eq(0))      // Shadow
    .and(qa.bitwiseAnd(1 << 4).eq(0));     // Snow
  return image.updateMask(mask);
};

// For each fire, download pre and post-fire satellite images, calculate burn severity (dNBR), and validate data quality
var processed = incidents
  .filter(ee.Filter.neq('attr_InitialLongitude', null))
  .filter(ee.Filter.neq('attr_InitialLatitude', null))
  .map(function(feature) {
    var lon = ee.Number(feature.get('attr_InitialLongitude'));
    var lat = ee.Number(feature.get('attr_InitialLatitude'));
    // ROI = 5km radius around fire center (used in: reduceRegion, filterBounds)
    var roi = ee.Geometry.Point([lon, lat]).buffer(5000).bounds();
    
    
    var preCol = ee.ImageCollection(COLLECTION_ID).filterBounds(roi)
                  .filterDate(feature.get('pre_start'), feature.get('pre_end'))
                  .filter(ee.Filter.lt('CLOUD_COVERAGE', CLOUD_THRESHOLD)).map(maskHLS); 
    var postCol = ee.ImageCollection(COLLECTION_ID).filterBounds(roi)
                   .filterDate(feature.get('post_start'), feature.get('post_end'))
                   .filter(ee.Filter.lt('CLOUD_COVERAGE', CLOUD_THRESHOLD)).map(maskHLS);

    var hasData = preCol.size().gt(0).and(postCol.size().gt(0));
    
    var analysis = ee.Dictionary(ee.Algorithms.If(hasData, 
      (function() {
        var preLand  = preCol.median().updateMask(waterMask);
        var postLand = postCol.median().updateMask(waterMask);

        // Calculate what fraction of the ROI has valid (unmasked) satellite data
        // This is critical: masks remove clouds, shadows, snow → only keep legitimate pixels
        var preValid  = ee.Number(preLand.select(0).reduceRegion({
          reducer: ee.Reducer.count(), geometry: roi, scale: 30, maxPixels: 1e9
        }).values().get(0));  // Count how many valid pixels in pre-fire image
        
        var preTotal  = ee.Number(preLand.select(0).unmask(0).reduceRegion({
          reducer: ee.Reducer.count(), geometry: roi, scale: 30, maxPixels: 1e9
        }).values().get(0));  // Count total pixels in ROI (including masked out)
        
        var postValid = ee.Number(postLand.select(0).reduceRegion({
          reducer: ee.Reducer.count(), geometry: roi, scale: 30, maxPixels: 1e9
        }).values().get(0));  // Count valid pixels in post-fire image
        
        var postTotal = ee.Number(postLand.select(0).unmask(0).reduceRegion({
          reducer: ee.Reducer.count(), geometry: roi, scale: 30, maxPixels: 1e9
        }).values().get(0));  // Count total pixels in ROI

        // Coverage check: valid pixels / total pixels in ROI
        var preCoverage  = preValid.divide(preTotal);
        var postCoverage = postValid.divide(postTotal);
        var hasCoverage  = preCoverage.gte(MIN_VALID_COVERAGE).and(postCoverage.gte(MIN_VALID_COVERAGE));

        // Calculate dNBR = pre NBR - post NBR, where NBR = (B5 NIR - B7 SWIR2) / (B5 NIR + B7 SWIR2)
        var d = preLand.normalizedDifference(['B5', 'B7']).subtract(postLand.normalizedDifference(['B5', 'B7'])).rename('dnbr');

        // Metric 1: 90th percentile dNBR (used in: safeFraction.gte(MIN_MODERATE_BURN_FRACTION))
        var val = d.reduceRegion({reducer: ee.Reducer.percentile([90]), geometry: roi, scale: 30, maxPixels: 1e9}).get('dnbr');
        
        // Metric 2: Fraction with dNBR > 0.4 (used in: safeFraction.gte(MIN_MODERATE_BURN_FRACTION))
        var moderateBurnFraction = d.gt(0.4).reduceRegion({reducer: ee.Reducer.mean(), geometry: roi, scale: 30, maxPixels: 1e9}).get('dnbr');

        // Convert to safe numbers: if calculation failed, return -999 (indicates bad data)
        var safeVal      = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(val, null),                  ee.Number(-999), val));
        var safeFraction = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(moderateBurnFraction, null), ee.Number(-999), moderateBurnFraction));

        // Validate: hasCoverage AND safe values AND moderate burn threshold
        var isValid = hasCoverage.and(safeVal.gt(0)).and(safeFraction.gte(MIN_MODERATE_BURN_FRACTION));

        return ee.Dictionary(ee.Algorithms.If(isValid,
          {v: safeVal, valid: 1, moderate_burn_fraction: safeFraction, pre_coverage: preCoverage, post_coverage: postCoverage},
          {v: ee.Number(-999), valid: 0, moderate_burn_fraction: safeFraction, pre_coverage: preCoverage, post_coverage: postCoverage}
        ));
      })(), 
      {v: ee.Number(-999), valid: 0, moderate_burn_fraction: ee.Number(-999), pre_coverage: ee.Number(-999), post_coverage: ee.Number(-999)}
    ));

    return feature.set({
      'severity_90th':          analysis.get('v'), 
      'moderate_burn_fraction': analysis.get('moderate_burn_fraction'),
      'pre_coverage':           analysis.get('pre_coverage'),
      'post_coverage':          analysis.get('post_coverage'),
      'has_data':               analysis.get('valid')
    });
}).filter(ee.Filter.eq('has_data', 1));  // Keep only fires that pass validation

// Selection: Sort by severity and take top TARGET_COUNT
var allValidated = processed.sort('severity_90th', false);  // Sort highest severity first
var diverseSet = allValidated.limit(TARGET_COUNT);  // Take top 50 (used in: Export.table.toDrive)

// Print validation results to GEE console with quality metrics for each fire
diverseSet.evaluate(function(featureList, error) {
  if (error || !featureList) {
    print('Server Error or No Sites Found:', error);
    return;
  }
  
  print('═══════════════════════════════════════════════════════════════════════════════════');
  print('EXPORT: ' + featureList.features.length + ' validated sites to Google Drive');
  print('═══════════════════════════════════════════════════════════════════════════════════');
  
  featureList.features.forEach(function(f, index) {
    var p = f.properties;
    var name = p.attr_IncidentName;
    var roi = ee.Geometry.Point([p.attr_InitialLongitude, p.attr_InitialLatitude]).buffer(5000).bounds();

    // Print: INDEX | COUNTRY | SEVERITY_90TH | BURN% | PRE/POST COVERAGE | DATES
    print((index + 1) + '. ' + name + ' | ' + p.country + 
          ' | Severity: ' + p.severity_90th.toFixed(4) +
          ' | Burn: ' + (p.moderate_burn_fraction * 100).toFixed(1) + '%' +
          ' | Pre/Post Coverage: ' + (p.pre_coverage * 100).toFixed(1) + '% / ' + (p.post_coverage * 100).toFixed(1) + '%' +
          ' | ' + p.IDate + ' to ' + p.FDate);

    // Fetch satellite imagery: pre and post fire (5km radius around fire center)
    var pre  = ee.ImageCollection(COLLECTION_ID).filterBounds(roi).filterDate(p.pre_start, p.pre_end)
                .filter(ee.Filter.lt('CLOUD_COVERAGE', CLOUD_THRESHOLD)).map(maskHLS)
                .median().updateMask(waterMask).clip(roi);  // Median composite of cloud-free images
    var post = ee.ImageCollection(COLLECTION_ID).filterBounds(roi).filterDate(p.post_start, p.post_end)
                .filter(ee.Filter.lt('CLOUD_COVERAGE', CLOUD_THRESHOLD)).map(maskHLS)
                .median().updateMask(waterMask).clip(roi);
    
    // Recalculate dNBR for visualization (shows burn severity spatially across ROI)
    var dnbr = pre.normalizedDifference(['B5', 'B7']).subtract(post.normalizedDifference(['B5', 'B7']));

    // Add RGB layers to map (for visual inspection of fire extent and surroundings)
    Map.addLayer(pre,  rgbVis,  name + ' 1: Pre-Fire (RGB)',  false);  // True color before fire
    Map.addLayer(post, rgbVis,  name + ' 2: Post-Fire (RGB)', false);  // True color after fire
    Map.addLayer(dnbr, dnbrVis, name + ' 3: dNBR Map (Severity)',  false);  // Burn severity heatmap
  });
});

// Export to Google Drive: fire names, locations, dates, satellite windows, quality metrics
Export.table.toDrive({
  collection: diverseSet,
  description: '50_Selected_sites',
  fileFormat: 'CSV',
  selectors: ['attr_IncidentName', 'country', 'grid_id', 'attr_InitialLongitude', 'attr_InitialLatitude',
              'IDate', 'FDate', 'area_ha',
              'pre_start', 'pre_end', 'post_start', 'post_end', 
              'severity_90th', 'moderate_burn_fraction', 'pre_coverage', 'post_coverage']
});
