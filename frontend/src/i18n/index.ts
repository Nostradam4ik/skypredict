import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { translations } from './translations';

// Get saved language or default to browser language
const getSavedLanguage = (): string => {
  const saved = localStorage.getItem('skypredict-language');
  if (saved && ['en', 'fr', 'es'].includes(saved)) {
    return saved;
  }

  // Try to detect browser language
  const browserLang = navigator.language.split('-')[0];
  if (['en', 'fr', 'es'].includes(browserLang)) {
    return browserLang;
  }

  return 'en';
};

i18n
  .use(initReactI18next)
  .init({
    resources: translations,
    lng: getSavedLanguage(),
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false
    }
  });

export default i18n;
