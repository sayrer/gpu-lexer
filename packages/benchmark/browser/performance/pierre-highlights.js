import { codeToTokens } from "@pierre/highlights";
import { githubLight } from "@pierre/highlights/themes";

// Themed JavaScript tokens, matching the Shiki worker's codeToTokens path.
const options = { lang: "javascript", theme: githubLight };

self.onmessage = ({ data: source }) => {
  try {
    codeToTokens(source, options);
    const started = performance.now();
    codeToTokens(source, options);
    self.postMessage({ elapsed: performance.now() - started });
  } catch (error) {
    self.postMessage({ error: error?.message ?? String(error) });
  }
};
