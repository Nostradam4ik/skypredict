export const translations = {
  en: {
    translation: {
      // Header
      title: 'SkyPredict',
      tagline: 'ML-Powered Flight Delay Prediction',
      apiConnected: 'API Connected',
      apiDisconnected: 'API Disconnected',
      checking: 'Checking...',
      compare: 'Compare',

      // Form
      flightDetails: 'Flight Details',
      airline: 'Airline',
      date: 'Date',
      origin: 'Origin',
      destination: 'Destination',
      departureTime: 'Departure Time',
      distance: 'Distance',
      estimatedFlightTime: 'Estimated flight time',
      addWeatherConditions: 'Add Weather Conditions',
      originWeather: 'Origin Weather',
      destWeather: 'Destination Weather',
      temp: 'Temp (°C)',
      wind: 'Wind (m/s)',
      visibility: 'Visibility (m)',
      rain: 'Rain (mm)',
      predictDelay: 'Predict Delay',
      predicting: 'Predicting...',
      addToCompare: 'Add to Compare',
      searchAirline: 'Search airline...',
      searchAirport: 'Search airport...',

      // Results
      onTime: 'ON TIME',
      delayed: 'DELAYED',
      likelyDelayed: 'LIKELY DELAYED',
      lowRisk: 'LOW RISK',
      mediumRisk: 'MEDIUM RISK',
      highRisk: 'HIGH RISK',
      severeRisk: 'SEVERE RISK',
      delayProbability: 'Delay Probability',
      estimatedDelay: 'Estimated Delay',
      confidence: 'Confidence',
      exportPdf: 'Export PDF',
      whyPrediction: 'Why this prediction?',

      // Risk Messages
      riskLow: 'Low delay risk. Your flight is expected to be on time.',
      riskMedium: 'Moderate delay risk. Monitor your flight status.',
      riskHigh: 'High delay risk. Consider checking alternatives.',
      riskSevere: 'Severe delay risk! Expect significant delays.',

      // History
      recentPredictions: 'Recent Predictions',
      clearHistory: 'Clear history',

      // Comparison
      flightComparison: 'Flight Comparison',
      clearAll: 'Clear All',
      bestOption: 'Best Option',
      delayRisk: 'Delay Risk',
      riskLevel: 'Risk Level',
      estDelay: 'Est. Delay',
      recommendation: 'Recommendation',
      lowestDelayRisk: 'has the lowest delay risk',

      // Stats
      analyticsDashboard: 'Analytics Dashboard',
      predictions: 'Predictions',
      avgRisk: 'Avg Risk',
      riskDistribution: 'Risk Distribution',
      hourlyTrend: 'Hourly Trend',
      topAirports: 'Top Airports',

      // Share
      share: 'Share',
      copyLink: 'Copy link',
      linkCopied: 'Link copied!',
      shareFlightPrediction: 'Share Flight Prediction',
      scanToView: 'Scan to view this prediction',
      downloadQrCode: 'Download QR Code',

      // Placeholder
      enterFlightDetails: 'Enter flight details',
      fillFormToPredict: 'Fill in the form to predict if your flight will be delayed',

      // Footer
      footerText: 'Built with XGBoost, FastAPI & React',
      trainedOn: 'Model trained on US domestic flight data',

      // Notifications
      highAlert: 'High Alert',
      warning: 'Warning',
      moderateRisk: 'Moderate risk',
      goodNews: 'Good news',
      likelyOnTime: 'is likely on time',
      addedToComparison: 'Added to comparison',
      maxFlightsCompare: 'Maximum 4 flights can be compared at once',
      failedPrediction: 'Failed to get prediction. Please try again.',
      historyCleared: 'History cleared',

      // Keyboard shortcuts
      keyboardShortcuts: 'Keyboard Shortcuts',
      pressToPredict: 'Press Enter to predict',
      pressToToggleTheme: 'Toggle theme',
      pressToCompare: 'Toggle compare mode',
      pressToFullscreen: 'Fullscreen result',
      pressToClose: 'Close modal',

      // Additional
      showStats: 'Show Analytics',
      hideStats: 'Hide Analytics',
      fullscreen: 'Fullscreen',
      qrCode: 'QR Code'
    }
  },
  fr: {
    translation: {
      // Header
      title: 'SkyPredict',
      tagline: 'Prédiction de Retards de Vols par ML',
      apiConnected: 'API Connectée',
      apiDisconnected: 'API Déconnectée',
      checking: 'Vérification...',
      compare: 'Comparer',

      // Form
      flightDetails: 'Détails du Vol',
      airline: 'Compagnie',
      date: 'Date',
      origin: 'Origine',
      destination: 'Destination',
      departureTime: 'Heure de Départ',
      distance: 'Distance',
      estimatedFlightTime: 'Durée estimée du vol',
      addWeatherConditions: 'Ajouter Conditions Météo',
      originWeather: 'Météo Origine',
      destWeather: 'Météo Destination',
      temp: 'Temp (°C)',
      wind: 'Vent (m/s)',
      visibility: 'Visibilité (m)',
      rain: 'Pluie (mm)',
      predictDelay: 'Prédire le Retard',
      predicting: 'Analyse...',
      addToCompare: 'Ajouter à la comparaison',
      searchAirline: 'Rechercher compagnie...',
      searchAirport: 'Rechercher aéroport...',

      // Results
      onTime: 'À L\'HEURE',
      delayed: 'EN RETARD',
      likelyDelayed: 'RETARD PROBABLE',
      lowRisk: 'RISQUE FAIBLE',
      mediumRisk: 'RISQUE MODÉRÉ',
      highRisk: 'RISQUE ÉLEVÉ',
      severeRisk: 'RISQUE SÉVÈRE',
      delayProbability: 'Probabilité de Retard',
      estimatedDelay: 'Retard Estimé',
      confidence: 'Confiance',
      exportPdf: 'Exporter PDF',
      whyPrediction: 'Pourquoi cette prédiction ?',

      // Risk Messages
      riskLow: 'Faible risque de retard. Votre vol devrait être à l\'heure.',
      riskMedium: 'Risque modéré de retard. Surveillez le statut de votre vol.',
      riskHigh: 'Risque élevé de retard. Envisagez de vérifier les alternatives.',
      riskSevere: 'Risque sévère de retard ! Attendez-vous à des retards importants.',

      // History
      recentPredictions: 'Prédictions Récentes',
      clearHistory: 'Effacer l\'historique',

      // Comparison
      flightComparison: 'Comparaison de Vols',
      clearAll: 'Tout Effacer',
      bestOption: 'Meilleure Option',
      delayRisk: 'Risque de Retard',
      riskLevel: 'Niveau de Risque',
      estDelay: 'Retard Est.',
      recommendation: 'Recommandation',
      lowestDelayRisk: 'a le plus faible risque de retard',

      // Stats
      analyticsDashboard: 'Tableau de Bord Analytics',
      predictions: 'Prédictions',
      avgRisk: 'Risque Moy.',
      riskDistribution: 'Distribution des Risques',
      hourlyTrend: 'Tendance Horaire',
      topAirports: 'Top Aéroports',

      // Share
      share: 'Partager',
      copyLink: 'Copier le lien',
      linkCopied: 'Lien copié !',
      shareFlightPrediction: 'Partager la Prédiction',
      scanToView: 'Scannez pour voir cette prédiction',
      downloadQrCode: 'Télécharger le QR Code',

      // Placeholder
      enterFlightDetails: 'Entrez les détails du vol',
      fillFormToPredict: 'Remplissez le formulaire pour prédire si votre vol sera retardé',

      // Footer
      footerText: 'Construit avec XGBoost, FastAPI & React',
      trainedOn: 'Modèle entraîné sur les données de vols domestiques US',

      // Notifications
      highAlert: 'Alerte Haute',
      warning: 'Attention',
      moderateRisk: 'Risque modéré',
      goodNews: 'Bonne nouvelle',
      likelyOnTime: 'devrait être à l\'heure',
      addedToComparison: 'Ajouté à la comparaison',
      maxFlightsCompare: 'Maximum 4 vols peuvent être comparés',
      failedPrediction: 'Échec de la prédiction. Veuillez réessayer.',
      historyCleared: 'Historique effacé',

      // Keyboard shortcuts
      keyboardShortcuts: 'Raccourcis Clavier',
      pressToPredict: 'Appuyez sur Entrée pour prédire',
      pressToToggleTheme: 'Changer le thème',
      pressToCompare: 'Mode comparaison',
      pressToFullscreen: 'Plein écran',
      pressToClose: 'Fermer',

      // Additional
      showStats: 'Afficher Statistiques',
      hideStats: 'Masquer Statistiques',
      fullscreen: 'Plein Écran',
      qrCode: 'Code QR'
    }
  },
  es: {
    translation: {
      // Header
      title: 'SkyPredict',
      tagline: 'Predicción de Retrasos de Vuelos con ML',
      apiConnected: 'API Conectada',
      apiDisconnected: 'API Desconectada',
      checking: 'Verificando...',
      compare: 'Comparar',

      // Form
      flightDetails: 'Detalles del Vuelo',
      airline: 'Aerolínea',
      date: 'Fecha',
      origin: 'Origen',
      destination: 'Destino',
      departureTime: 'Hora de Salida',
      distance: 'Distancia',
      estimatedFlightTime: 'Tiempo de vuelo estimado',
      addWeatherConditions: 'Agregar Condiciones Climáticas',
      originWeather: 'Clima Origen',
      destWeather: 'Clima Destino',
      temp: 'Temp (°C)',
      wind: 'Viento (m/s)',
      visibility: 'Visibilidad (m)',
      rain: 'Lluvia (mm)',
      predictDelay: 'Predecir Retraso',
      predicting: 'Prediciendo...',
      addToCompare: 'Agregar a comparación',
      searchAirline: 'Buscar aerolínea...',
      searchAirport: 'Buscar aeropuerto...',

      // Results
      onTime: 'A TIEMPO',
      delayed: 'RETRASADO',
      likelyDelayed: 'PROBABLE RETRASO',
      lowRisk: 'RIESGO BAJO',
      mediumRisk: 'RIESGO MEDIO',
      highRisk: 'RIESGO ALTO',
      severeRisk: 'RIESGO SEVERO',
      delayProbability: 'Probabilidad de Retraso',
      estimatedDelay: 'Retraso Estimado',
      confidence: 'Confianza',
      exportPdf: 'Exportar PDF',
      whyPrediction: '¿Por qué esta predicción?',

      // Risk Messages
      riskLow: 'Bajo riesgo de retraso. Su vuelo debería llegar a tiempo.',
      riskMedium: 'Riesgo moderado de retraso. Monitoree el estado de su vuelo.',
      riskHigh: 'Alto riesgo de retraso. Considere verificar alternativas.',
      riskSevere: '¡Riesgo severo de retraso! Espere retrasos significativos.',

      // History
      recentPredictions: 'Predicciones Recientes',
      clearHistory: 'Borrar historial',

      // Comparison
      flightComparison: 'Comparación de Vuelos',
      clearAll: 'Borrar Todo',
      bestOption: 'Mejor Opción',
      delayRisk: 'Riesgo de Retraso',
      riskLevel: 'Nivel de Riesgo',
      estDelay: 'Retraso Est.',
      recommendation: 'Recomendación',
      lowestDelayRisk: 'tiene el menor riesgo de retraso',

      // Stats
      analyticsDashboard: 'Panel de Análisis',
      predictions: 'Predicciones',
      avgRisk: 'Riesgo Prom.',
      riskDistribution: 'Distribución de Riesgos',
      hourlyTrend: 'Tendencia Horaria',
      topAirports: 'Top Aeropuertos',

      // Share
      share: 'Compartir',
      copyLink: 'Copiar enlace',
      linkCopied: '¡Enlace copiado!',
      shareFlightPrediction: 'Compartir Predicción',
      scanToView: 'Escanea para ver esta predicción',
      downloadQrCode: 'Descargar Código QR',

      // Placeholder
      enterFlightDetails: 'Ingrese detalles del vuelo',
      fillFormToPredict: 'Complete el formulario para predecir si su vuelo se retrasará',

      // Footer
      footerText: 'Construido con XGBoost, FastAPI & React',
      trainedOn: 'Modelo entrenado con datos de vuelos domésticos de EE.UU.',

      // Notifications
      highAlert: 'Alerta Alta',
      warning: 'Advertencia',
      moderateRisk: 'Riesgo moderado',
      goodNews: 'Buenas noticias',
      likelyOnTime: 'probablemente llegará a tiempo',
      addedToComparison: 'Agregado a la comparación',
      maxFlightsCompare: 'Máximo 4 vuelos pueden compararse',
      failedPrediction: 'Error en la predicción. Intente de nuevo.',
      historyCleared: 'Historial borrado',

      // Keyboard shortcuts
      keyboardShortcuts: 'Atajos de Teclado',
      pressToPredict: 'Presione Enter para predecir',
      pressToToggleTheme: 'Cambiar tema',
      pressToCompare: 'Modo comparación',
      pressToFullscreen: 'Pantalla completa',
      pressToClose: 'Cerrar',

      // Additional
      showStats: 'Mostrar Estadísticas',
      hideStats: 'Ocultar Estadísticas',
      fullscreen: 'Pantalla Completa',
      qrCode: 'Código QR'
    }
  }
};
