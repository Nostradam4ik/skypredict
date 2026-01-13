import React from 'react';
import { Twitter, Linkedin, Link2, Check } from 'lucide-react';
import { PredictionResponse } from '../types';

interface ShareButtonsProps {
  prediction: PredictionResponse;
}

const ShareButtons: React.FC<ShareButtonsProps> = ({ prediction }) => {
  const [copied, setCopied] = React.useState(false);

  const getShareText = () => {
    const prob = Math.round(prediction.delay_probability * 100);
    const status = prediction.is_delayed ? 'may be delayed' : 'is likely on time';
    return `My flight ${prediction.origin_airport} → ${prediction.dest_airport} ${status} (${prob}% delay risk). Predicted by SkyPredict! #FlightDelay #Travel`;
  };

  const getShareUrl = () => {
    // Create a shareable URL with flight details
    const params = new URLSearchParams({
      origin: prediction.origin_airport,
      dest: prediction.dest_airport,
      airline: prediction.airline,
      date: prediction.flight_date,
      risk: prediction.risk_level
    });
    return `${window.location.origin}?${params.toString()}`;
  };

  const shareOnTwitter = () => {
    const text = encodeURIComponent(getShareText());
    const url = encodeURIComponent(getShareUrl());
    window.open(`https://twitter.com/intent/tweet?text=${text}&url=${url}`, '_blank');
  };

  const shareOnLinkedIn = () => {
    const url = encodeURIComponent(getShareUrl());
    const title = encodeURIComponent(`Flight Delay Prediction: ${prediction.origin_airport} → ${prediction.dest_airport}`);
    const summary = encodeURIComponent(getShareText());
    window.open(`https://www.linkedin.com/shareArticle?mini=true&url=${url}&title=${title}&summary=${summary}`, '_blank');
  };

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(getShareUrl());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy link');
    }
  };

  return (
    <div className="share-buttons">
      <span className="share-label">Share:</span>
      <button
        className="share-btn twitter"
        onClick={shareOnTwitter}
        title="Share on Twitter"
      >
        <Twitter size={16} />
      </button>
      <button
        className="share-btn linkedin"
        onClick={shareOnLinkedIn}
        title="Share on LinkedIn"
      >
        <Linkedin size={16} />
      </button>
      <button
        className="share-btn copy"
        onClick={copyLink}
        title="Copy link"
      >
        {copied ? <Check size={16} /> : <Link2 size={16} />}
      </button>
    </div>
  );
};

export default ShareButtons;
