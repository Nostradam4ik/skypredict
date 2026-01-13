import React from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { Download, X } from 'lucide-react';
import { PredictionResponse } from '../types';

interface QRCodeShareProps {
  prediction: PredictionResponse;
  onClose: () => void;
}

const QRCodeShare: React.FC<QRCodeShareProps> = ({ prediction, onClose }) => {
  const getShareUrl = () => {
    const params = new URLSearchParams({
      origin: prediction.origin_airport,
      dest: prediction.dest_airport,
      airline: prediction.airline,
      date: prediction.flight_date,
      risk: prediction.risk_level
    });
    return `${window.location.origin}?${params.toString()}`;
  };

  const downloadQR = () => {
    const svg = document.getElementById('qr-code-svg');
    if (!svg) return;

    const svgData = new XMLSerializer().serializeToString(svg);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx?.drawImage(img, 0, 0);
      const pngFile = canvas.toDataURL('image/png');
      const downloadLink = document.createElement('a');
      downloadLink.download = `skypredict_${prediction.origin_airport}_${prediction.dest_airport}.png`;
      downloadLink.href = pngFile;
      downloadLink.click();
    };

    img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
  };

  return (
    <div className="qr-modal-overlay" onClick={onClose}>
      <div className="qr-modal" onClick={e => e.stopPropagation()}>
        <button className="qr-close" onClick={onClose}>
          <X size={20} />
        </button>

        <h3>Share Flight Prediction</h3>
        <p className="qr-route">
          {prediction.origin_airport} → {prediction.dest_airport}
        </p>

        <div className="qr-code-wrapper">
          <QRCodeSVG
            id="qr-code-svg"
            value={getShareUrl()}
            size={200}
            level="H"
            includeMargin
            bgColor="white"
            fgColor="#1e293b"
          />
        </div>

        <p className="qr-hint">Scan to view this prediction</p>

        <button className="qr-download-btn" onClick={downloadQR}>
          <Download size={18} />
          Download QR Code
        </button>
      </div>
    </div>
  );
};

export default QRCodeShare;
