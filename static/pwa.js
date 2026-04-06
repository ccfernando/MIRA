const INSTALL_HELP_PARAM_MAP = {
  iosSafari: "ios-safari",
  iosChrome: "ios-chrome",
  androidChrome: "android-chrome",
  fallback: "fallback",
};

let deferredInstallPrompt = null;

function isStandaloneMode() {
  return window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
}

function getUserAgent() {
  return window.navigator.userAgent;
}

function isIosDevice() {
  const userAgent = getUserAgent();
  return /iPad|iPhone|iPod/.test(userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
}

function isAndroidDevice() {
  return /Android/i.test(getUserAgent());
}

function isIosSafari() {
  const userAgent = getUserAgent();
  const isWebkitBrowser = /WebKit/.test(userAgent);
  const isOtherIosBrowser = /CriOS|FxiOS|EdgiOS|OPiOS/.test(userAgent);
  return isIosDevice() && isWebkitBrowser && !isOtherIosBrowser;
}

function isIosChrome() {
  return isIosDevice() && /CriOS/.test(getUserAgent());
}

function isAndroidChrome() {
  const userAgent = getUserAgent();
  return isAndroidDevice() && /Chrome/i.test(userAgent) && !/EdgA|OPR|SamsungBrowser/i.test(userAgent);
}

function getInstallHelpUrl(source) {
  const button = document.querySelector("[data-install-help-url]");
  const helpUrl = button?.dataset.installHelpUrl || "/install";
  const url = new URL(helpUrl, window.location.origin);
  url.searchParams.set("source", INSTALL_HELP_PARAM_MAP[source] || INSTALL_HELP_PARAM_MAP.fallback);
  return url.toString();
}

function getInstallMode() {
  if (isStandaloneMode()) {
    return "installed";
  }

  if (deferredInstallPrompt !== null) {
    return "prompt";
  }

  if (isIosSafari()) {
    return "ios-safari";
  }

  if (isIosChrome()) {
    return "ios-chrome";
  }

  if (isAndroidChrome()) {
    return "android-chrome-manual";
  }

  return "fallback";
}

function updateInstallButtons() {
  const buttons = document.querySelectorAll("[data-install-app]");
  const installMode = getInstallMode();

  buttons.forEach((button) => {
    button.hidden = installMode === "installed";
    button.textContent = "Install App";
    button.dataset.installMode = installMode;
  });
}

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  deferredInstallPrompt = event;
  updateInstallButtons();
});

window.addEventListener("appinstalled", () => {
  deferredInstallPrompt = null;
  updateInstallButtons();
});

document.addEventListener("click", async (event) => {
  const installButton = event.target.closest("[data-install-app]");
  if (!installButton) {
    return;
  }

  const installMode = getInstallMode();

  if (installMode === "prompt" && deferredInstallPrompt !== null) {
    deferredInstallPrompt.prompt();
    await deferredInstallPrompt.userChoice;
    deferredInstallPrompt = null;
    updateInstallButtons();
    return;
  }

  if (installMode === "ios-safari") {
    window.location.href = getInstallHelpUrl("iosSafari");
    return;
  }

  if (installMode === "ios-chrome") {
    window.location.href = getInstallHelpUrl("iosChrome");
    return;
  }

  if (installMode === "android-chrome-manual") {
    window.location.href = getInstallHelpUrl("androidChrome");
    return;
  }

  window.location.href = getInstallHelpUrl("fallback");
});

window.addEventListener("load", async () => {
  if ("serviceWorker" in navigator) {
    try {
      const registration = await navigator.serviceWorker.register("/service-worker.js");
      registration.update();
    } catch (error) {
      console.warn("Service worker registration failed", error);
    }
  }

  updateInstallButtons();
});

window.addEventListener("resize", updateInstallButtons);
