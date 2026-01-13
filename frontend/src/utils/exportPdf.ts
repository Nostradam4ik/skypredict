import jsPDF from 'jspdf';
import { PredictionResponse } from '../types';

export const exportToPdf = async (prediction: PredictionResponse) => {
  const pdf = new jsPDF();
  const pageWidth = pdf.internal.pageSize.getWidth();

  // Colors as tuples
  const primaryColor: [number, number, number] = [59, 130, 246];
  const textColor: [number, number, number] = [30, 41, 59];
  const secondaryColor: [number, number, number] = [148, 163, 184];

  const getRiskColor = (risk: string): [number, number, number] => {
    switch (risk) {
      case 'LOW': return [34, 197, 94];
      case 'MEDIUM': return [245, 158, 11];
      case 'HIGH': return [239, 68, 68];
      case 'SEVERE': return [124, 45, 18];
      default: return [148, 163, 184];
    }
  };

  // Header
  pdf.setFillColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  pdf.rect(0, 0, pageWidth, 40, 'F');

  pdf.setTextColor(255, 255, 255);
  pdf.setFontSize(24);
  pdf.setFont('helvetica', 'bold');
  pdf.text('SkyPredict', 20, 25);

  pdf.setFontSize(10);
  pdf.setFont('helvetica', 'normal');
  pdf.text('Flight Delay Prediction Report', 20, 33);

  // Date
  pdf.setFontSize(10);
  pdf.text(`Generated: ${new Date().toLocaleString()}`, pageWidth - 70, 25);

  // Main content
  let y = 55;

  // Flight Route Section
  pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
  pdf.setFontSize(16);
  pdf.setFont('helvetica', 'bold');
  pdf.text('Flight Details', 20, y);
  y += 15;

  // Route box
  pdf.setFillColor(241, 245, 249);
  pdf.roundedRect(20, y, pageWidth - 40, 35, 3, 3, 'F');

  pdf.setFontSize(24);
  pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
  pdf.text(prediction.origin_airport, 40, y + 22);
  pdf.text('→', pageWidth / 2 - 5, y + 22);
  pdf.text(prediction.dest_airport, pageWidth - 65, y + 22);

  y += 45;

  // Flight info
  pdf.setFontSize(11);
  pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
  const flightInfo = [
    `Airline: ${prediction.airline}`,
    `Date: ${prediction.flight_date}`,
    `Departure: ${prediction.scheduled_departure}`
  ];
  flightInfo.forEach((info, index) => {
    pdf.text(info, 20 + (index * 60), y);
  });

  y += 20;

  // Prediction Results Section
  pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
  pdf.setFontSize(16);
  pdf.setFont('helvetica', 'bold');
  pdf.text('Prediction Results', 20, y);
  y += 15;

  // Risk level box
  const riskColor = getRiskColor(prediction.risk_level);
  pdf.setFillColor(riskColor[0], riskColor[1], riskColor[2]);
  pdf.roundedRect(20, y, pageWidth - 40, 50, 3, 3, 'F');

  pdf.setTextColor(255, 255, 255);
  pdf.setFontSize(28);
  pdf.setFont('helvetica', 'bold');
  const statusText = prediction.is_delayed ? 'LIKELY DELAYED' : 'ON TIME';
  pdf.text(statusText, pageWidth / 2, y + 25, { align: 'center' });

  pdf.setFontSize(12);
  pdf.setFont('helvetica', 'normal');
  pdf.text(`Risk Level: ${prediction.risk_level}`, pageWidth / 2, y + 40, { align: 'center' });

  y += 65;

  // Statistics
  pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
  pdf.setFontSize(12);

  const stats = [
    { label: 'Delay Probability', value: `${Math.round(prediction.delay_probability * 100)}%` },
    { label: 'Estimated Delay', value: prediction.estimated_delay_minutes ? `${prediction.estimated_delay_minutes} min` : 'N/A' },
    { label: 'Model Confidence', value: `${Math.round(prediction.confidence * 100)}%` }
  ];

  const statWidth = (pageWidth - 60) / 3;
  stats.forEach((stat, index) => {
    const x = 20 + (index * statWidth) + (statWidth / 2);

    pdf.setFillColor(241, 245, 249);
    pdf.roundedRect(20 + (index * statWidth) + 5, y, statWidth - 10, 40, 3, 3, 'F');

    pdf.setFontSize(10);
    pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
    pdf.text(stat.label, x, y + 15, { align: 'center' });

    pdf.setFontSize(18);
    pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
    pdf.setFont('helvetica', 'bold');
    pdf.text(stat.value, x, y + 30, { align: 'center' });
  });

  y += 55;

  // Risk Message
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(11);
  pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);

  const riskMessages: Record<string, string> = {
    LOW: 'Low risk of delay. Your flight is expected to depart on time.',
    MEDIUM: 'Moderate delay risk. Monitor your flight status for updates.',
    HIGH: 'High risk of delay. Consider arriving early and checking alternatives.',
    SEVERE: 'Very high risk of significant delay. Plan for contingencies.'
  };

  const message = riskMessages[prediction.risk_level] || '';
  pdf.text(message, 20, y, { maxWidth: pageWidth - 40 });

  y += 20;

  // Feature explanations if available
  if (prediction.explanation?.features && prediction.explanation.features.length > 0) {
    y += 10;
    pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
    pdf.setFontSize(14);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Key Factors', 20, y);
    y += 10;

    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');

    prediction.explanation.features.slice(0, 5).forEach((feature) => {
      const impact = feature.shap_value > 0 ? '+' : '-';
      const impactColor: [number, number, number] = feature.shap_value > 0 ? [239, 68, 68] : [34, 197, 94];

      pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
      pdf.text(`${feature.name}: ${feature.value}`, 25, y);

      pdf.setTextColor(impactColor[0], impactColor[1], impactColor[2]);
      pdf.text(`${impact} ${feature.impact}`, pageWidth - 50, y);

      y += 8;
    });
  }

  // Footer
  pdf.setFillColor(241, 245, 249);
  pdf.rect(0, pdf.internal.pageSize.getHeight() - 20, pageWidth, 20, 'F');

  pdf.setFontSize(8);
  pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
  pdf.text(
    'SkyPredict - ML-Powered Flight Delay Prediction | This is a prediction and not a guarantee.',
    pageWidth / 2,
    pdf.internal.pageSize.getHeight() - 8,
    { align: 'center' }
  );

  // Save
  const filename = `skypredict_${prediction.origin_airport}_${prediction.dest_airport}_${prediction.flight_date}.pdf`;
  pdf.save(filename);
};
