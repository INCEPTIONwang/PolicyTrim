const copyButton = document.querySelector("[data-copy-bib]");
const navLinks = [...document.querySelectorAll(".nav-links a")];
const videoFrames = [...document.querySelectorAll(".video-frame")];
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

if (copyButton) {
  copyButton.addEventListener("click", async () => {
    const bibtex = document.querySelector(".bibtex code")?.textContent.trim();
    if (!bibtex) return;

    await navigator.clipboard.writeText(bibtex);
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy";
    }, 1600);
  });
}

function updateActiveNav() {
  let activeId = "";
  for (const section of sections) {
    if (window.scrollY >= section.offsetTop - 110) {
      activeId = section.id;
    }
  }

  for (const link of navLinks) {
    link.classList.toggle("active", link.getAttribute("href") === `#${activeId}`);
  }
}

window.addEventListener("scroll", updateActiveNav, { passive: true });
updateActiveNav();

function formatVideoTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00.0";

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds - minutes * 60;

  return `${String(minutes).padStart(2, "0")}:${remainingSeconds.toFixed(1).padStart(4, "0")}`;
}

for (const frame of videoFrames) {
  const video = frame.querySelector("video");
  const timeLabel = frame.querySelector("[data-video-time]");
  if (!video || !timeLabel) continue;

  const successTime = Number.parseFloat(video.dataset.successTime || "0");

  function syncVideoOverlay() {
    timeLabel.textContent = formatVideoTime(video.currentTime);
    frame.classList.toggle("is-success", video.currentTime + 0.05 >= successTime);
  }

  video.addEventListener("loadedmetadata", syncVideoOverlay);
  video.addEventListener("timeupdate", syncVideoOverlay);
  video.addEventListener("seeked", syncVideoOverlay);
  video.addEventListener("ended", syncVideoOverlay);
  syncVideoOverlay();
}
