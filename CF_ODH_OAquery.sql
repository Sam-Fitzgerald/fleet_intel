SELECT [ScadamSiteID]
      ,t1.[Site]
      ,t1.[StartDate]
	  ,t1.[TurbineName]
	  ,t2.[TurbineApiKey]
      ,t1.[Production]
      ,t1.[OperationalLostProduction]
      ,t1.[CapacityFactor]
      ,t1.[OperationalDowntimeHours]
      ,t1.[OperationalAvailability]
      ,t1.[WindSpeedMean]
      ,t1.[WindSpeedStandardDeviation]
	  ,t1.[TemperatureMean]
	  ,t1.[TurbulenceMean]
	  ,t1.[TemperatureStandardDeviation]
	  ,t1.[WindFarmAvailability]
	  ,t1.[TurbineAvailability]
FROM [dbo].[vwMonthlyPerTurbineResults] t1
LEFT JOIN 
    [dbo].[vwSiteTurbines] t2 
    ON T1.[TurbineName] = T2.TurbineName
WHERE
    t1.[StartDate] >= '2010-01-01' 
    AND COALESCE([CapacityFactor],[OperationalDowntimeHours],[OperationalAvailability]) IS NOT NULL
    AND ProductionDataCoverage > 0.8
ORDER BY
    t1.[Site]
    ,[TurbineName]
    ,[StartDate]