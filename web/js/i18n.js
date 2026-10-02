// Tiny i18n layer (English / French). Static text uses data-i18n* attributes,
// dynamic text goes through t().

import { COCO_CLASSES } from './config.js';

export const STRINGS = {
  en: {
    'meta.title': 'BlurAnything — Blur anything in your photos, privately',
    'meta.description':
      'Detect and blur people, vehicles, animals and objects in your photos. 100% in your browser with YOLOv8 and WebAssembly — nothing is uploaded.',
    skip: 'Skip to content',
    'hero.badge': '100% in your browser · nothing is uploaded',
    'hero.title': 'Blur anything in your photos.',
    'hero.lead':
      'AI finds people, vehicles, animals and everyday objects. You pick what to hide. Your images never leave your device.',
    'drop.title': 'Drop an image here',
    'drop.hint': 'or click to browse — you can also paste with Ctrl+V',
    'drop.formats': 'JPG, PNG, WebP…',
    'drop.overlay': 'Drop to open',
    'samples.title': 'No image at hand? Try an example',
    'sample.bus': 'Street',
    'sample.astronaut': 'Portrait',
    'sample.cat': 'Cat',
    'sample.coffee': 'Coffee',
    'feat.private.title': 'Private by design',
    'feat.private.text':
      'The AI model runs in your browser with WebAssembly. No upload, no account, no tracking — and it works offline once loaded.',
    'feat.precise.title': 'Precise masks',
    'feat.precise.text': 'Instance segmentation follows the outline of each object instead of a crude rectangle.',
    'feat.anything.title': 'Blur anything',
    'feat.anything.text': 'Click detected objects, or draw your own areas for faces, plates, screens and text.',
    'tool.rect': 'Rectangle',
    'tool.rect.title': 'Draw a rectangular area to blur',
    'tool.ellipse': 'Ellipse',
    'tool.ellipse.title': 'Draw an elliptical area to blur',
    'tool.compare': 'Hold to compare',
    'tool.compare.title': 'Hold to see the original',
    'tool.new': 'New image',
    'tool.download': 'Download',
    'tool.copy': 'Copy',
    'panel.model': 'Detection model',
    'model.fast': 'Fast',
    'model.fast.sub': 'YOLOv8n · {size}',
    'model.accurate': 'Accurate',
    'model.accurate.sub': 'YOLOv8s · {size}',
    'panel.objects': 'Objects',
    'objects.count.one': '{n} object',
    'objects.count.other': '{n} objects',
    'objects.all': 'All',
    'objects.none': 'None',
    'objects.empty': 'Nothing found. Lower the confidence or draw an area on the image.',
    'objects.conf': 'Minimum confidence',
    'panel.effect': 'Effect',
    'style.blur': 'Blur',
    'style.pixelate': 'Pixelate',
    'style.solid': 'Cover',
    'effect.strength': 'Strength',
    'effect.margin': 'Margin',
    'effect.margin.title': 'Grows the mask so outlines are fully covered',
    'region.custom': 'Area',
    'region.remove': 'Remove',
    'hint.default': 'Click an object on the image to blur it — or draw an area for anything else.',
    'hint.draw': 'Drag on the image to draw an area · Esc to cancel',
    'status.model': 'Loading model…',
    'status.modelPct': 'Downloading model… {pct}%',
    'status.detecting': 'Detecting objects…',
    'status.done': '{n} found in {ms} ms',
    'status.exporting': 'Preparing image…',
    'toast.copied': 'Copied to clipboard',
    'toast.copyFailed': "Couldn't copy — use Download instead",
    'toast.downscaled': 'Large image resized to {w}×{h}',
    'toast.saved': 'Image saved (metadata stripped)',
    'err.decode': "This file can't be read as an image.",
    'err.notImage': 'Please choose an image file.',
    'err.model': 'Could not load the AI model: {msg}',
    'err.detect': 'Detection failed: {msg}',
    'err.export': 'Export failed: {msg}',
    'err.unsupported':
      'Your browser lacks required features (WebAssembly, Web Workers). Try a recent Chrome, Edge, Firefox or Safari.',
    'action.retry': 'Retry',
    'footer.privacy': 'Everything runs locally. Exported images carry no EXIF/GPS metadata.',
    'footer.credits': 'Built with YOLOv8 and ONNX Runtime Web',
    'footer.source': 'Source code',
    'theme.toggle': 'Toggle dark mode',
    'lang.toggle': 'Passer en français',
    'a11y.region': '{name}, {pct}% confidence',
  },
  fr: {
    'meta.title': 'BlurAnything — Floutez n’importe quoi dans vos photos, en privé',
    'meta.description':
      'Détectez et floutez personnes, véhicules, animaux et objets dans vos photos. 100 % dans votre navigateur avec YOLOv8 et WebAssembly — rien n’est envoyé.',
    skip: 'Aller au contenu',
    'hero.badge': '100 % dans votre navigateur · rien n’est envoyé',
    'hero.title': 'Floutez n’importe quoi dans vos photos.',
    'hero.lead':
      'L’IA repère personnes, véhicules, animaux et objets du quotidien. Vous choisissez ce qu’il faut masquer. Vos images ne quittent jamais votre appareil.',
    'drop.title': 'Déposez une image ici',
    'drop.hint': 'ou cliquez pour parcourir — vous pouvez aussi coller avec Ctrl+V',
    'drop.formats': 'JPG, PNG, WebP…',
    'drop.overlay': 'Déposez pour ouvrir',
    'samples.title': 'Pas d’image sous la main ? Essayez un exemple',
    'sample.bus': 'Rue',
    'sample.astronaut': 'Portrait',
    'sample.cat': 'Chat',
    'sample.coffee': 'Café',
    'feat.private.title': 'Privé par conception',
    'feat.private.text':
      'Le modèle d’IA tourne dans votre navigateur avec WebAssembly. Aucun envoi, aucun compte, aucun suivi — et ça marche hors ligne une fois chargé.',
    'feat.precise.title': 'Masques précis',
    'feat.precise.text': 'La segmentation suit le contour de chaque objet au lieu d’un rectangle approximatif.',
    'feat.anything.title': 'Floutez tout',
    'feat.anything.text':
      'Cliquez sur les objets détectés, ou dessinez vos zones pour visages, plaques, écrans et textes.',
    'tool.rect': 'Rectangle',
    'tool.rect.title': 'Dessiner une zone rectangulaire à flouter',
    'tool.ellipse': 'Ellipse',
    'tool.ellipse.title': 'Dessiner une zone elliptique à flouter',
    'tool.compare': 'Maintenir pour comparer',
    'tool.compare.title': 'Maintenir pour voir l’original',
    'tool.new': 'Nouvelle image',
    'tool.download': 'Télécharger',
    'tool.copy': 'Copier',
    'panel.model': 'Modèle de détection',
    'model.fast': 'Rapide',
    'model.fast.sub': 'YOLOv8n · {size}',
    'model.accurate': 'Précis',
    'model.accurate.sub': 'YOLOv8s · {size}',
    'panel.objects': 'Objets',
    'objects.count.one': '{n} objet',
    'objects.count.other': '{n} objets',
    'objects.all': 'Tous',
    'objects.none': 'Aucun',
    'objects.empty': 'Rien trouvé. Baissez la confiance ou dessinez une zone sur l’image.',
    'objects.conf': 'Confiance minimale',
    'panel.effect': 'Effet',
    'style.blur': 'Flou',
    'style.pixelate': 'Pixeliser',
    'style.solid': 'Masquer',
    'effect.strength': 'Intensité',
    'effect.margin': 'Marge',
    'effect.margin.title': 'Agrandit le masque pour couvrir tout le contour',
    'region.custom': 'Zone',
    'region.remove': 'Supprimer',
    'hint.default': 'Cliquez sur un objet de l’image pour le flouter — ou dessinez une zone pour le reste.',
    'hint.draw': 'Glissez sur l’image pour dessiner une zone · Échap pour annuler',
    'status.model': 'Chargement du modèle…',
    'status.modelPct': 'Téléchargement du modèle… {pct} %',
    'status.detecting': 'Détection des objets…',
    'status.done': '{n} trouvé(s) en {ms} ms',
    'status.exporting': 'Préparation de l’image…',
    'toast.copied': 'Copié dans le presse-papiers',
    'toast.copyFailed': 'Copie impossible — utilisez Télécharger',
    'toast.downscaled': 'Grande image réduite à {w}×{h}',
    'toast.saved': 'Image enregistrée (métadonnées supprimées)',
    'err.decode': 'Ce fichier ne peut pas être lu comme une image.',
    'err.notImage': 'Veuillez choisir un fichier image.',
    'err.model': 'Impossible de charger le modèle d’IA : {msg}',
    'err.detect': 'Échec de la détection : {msg}',
    'err.export': 'Échec de l’export : {msg}',
    'err.unsupported':
      'Votre navigateur ne gère pas les fonctions requises (WebAssembly, Web Workers). Essayez un Chrome, Edge, Firefox ou Safari récent.',
    'action.retry': 'Réessayer',
    'footer.privacy': 'Tout s’exécute en local. Les images exportées n’ont aucune métadonnée EXIF/GPS.',
    'footer.credits': 'Construit avec YOLOv8 et ONNX Runtime Web',
    'footer.source': 'Code source',
    'theme.toggle': 'Basculer le mode sombre',
    'lang.toggle': 'Switch to English',
    'a11y.region': '{name}, confiance {pct} %',
  },
};

const FR_CLASSES = `personne, vélo, voiture, moto, avion, bus, train, camion,
bateau, feu tricolore, bouche d’incendie, panneau stop, parcmètre, banc, oiseau, chat,
chien, cheval, mouton, vache, éléphant, ours, zèbre, girafe,
sac à dos, parapluie, sac à main, cravate, valise, frisbee, skis, snowboard,
ballon, cerf-volant, batte de baseball, gant de baseball, skateboard, planche de surf, raquette de tennis, bouteille,
verre à vin, tasse, fourchette, couteau, cuillère, bol, banane, pomme,
sandwich, orange, brocoli, carotte, hot-dog, pizza, donut, gâteau,
chaise, canapé, plante en pot, lit, table, toilettes, télévision, ordinateur portable,
souris, télécommande, clavier, téléphone, micro-ondes, four, grille-pain, évier,
réfrigérateur, livre, horloge, vase, ciseaux, ours en peluche, sèche-cheveux, brosse à dents`.split(/,\s*/);

const CLASS_NAMES = { en: COCO_CLASSES, fr: FR_CLASSES };

let lang = 'en';

export function detectLang(preferred, navigatorLang) {
  if (preferred && STRINGS[preferred]) return preferred;
  return /^fr\b/i.test(navigatorLang || '') ? 'fr' : 'en';
}

export function getLang() {
  return lang;
}

export function setLang(next) {
  lang = STRINGS[next] ? next : 'en';
  if (typeof document !== 'undefined') document.documentElement.lang = lang;
}

export function t(key, vars) {
  const s = STRINGS[lang][key] ?? STRINGS.en[key] ?? key;
  return vars ? s.replace(/\{(\w+)\}/g, (_, k) => (k in vars ? vars[k] : `{${k}}`)) : s;
}

/** Plural-aware lookup: keys `<base>.one` / `<base>.other`. */
export function tn(base, n, vars = {}) {
  return t(`${base}.${n === 1 ? 'one' : 'other'}`, { n, ...vars });
}

export function className(classId) {
  return CLASS_NAMES[lang][classId] ?? COCO_CLASSES[classId] ?? String(classId);
}

export function formatBytes(bytes) {
  const mb = bytes / 1e6;
  return `${mb >= 10 ? Math.round(mb) : mb.toFixed(1)} ${lang === 'fr' ? 'Mo' : 'MB'}`;
}

/** Fill every [data-i18n*] element under `root`. */
export function applyI18n(root = document) {
  const set = (selector, attr, fn) => root.querySelectorAll(selector).forEach((el) => fn(el, t(el.getAttribute(attr))));
  set('[data-i18n]', 'data-i18n', (el, v) => (el.textContent = v));
  set('[data-i18n-title]', 'data-i18n-title', (el, v) => el.setAttribute('title', v));
  set('[data-i18n-label]', 'data-i18n-label', (el, v) => el.setAttribute('aria-label', v));
  set('[data-i18n-alt]', 'data-i18n-alt', (el, v) => el.setAttribute('alt', v));
  if (typeof document !== 'undefined') {
    document.title = t('meta.title');
    document.querySelector('meta[name="description"]')?.setAttribute('content', t('meta.description'));
  }
}
