import assert from "node:assert/strict";
import test from "node:test";

import { CLASSES } from "../yolo.js";
import { CLASS_FR, STRINGS } from "../i18n.js";

test("French and English define the same keys", () => {
  assert.deepEqual(Object.keys(STRINGS.fr).sort(), Object.keys(STRINGS.en).sort());
});

test("every COCO class has a French name", () => {
  assert.deepEqual(CLASSES.filter((c) => !CLASS_FR[c]), []);
});

test("plural forms", () => {
  assert.equal(STRINGS.fr.summary(1, 1), "1 objet détecté · 1 zone floutée");
  assert.equal(STRINGS.fr.summary(5, 0), "5 objets détectés · 0 zone floutée");
  assert.equal(STRINGS.en.summary(2, 2), "2 objects detected · 2 areas blurred");
});
