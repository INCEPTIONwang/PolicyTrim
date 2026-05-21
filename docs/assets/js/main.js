const copyButton = document.querySelector("[data-copy-bib]");
const navLinks = [...document.querySelectorAll(".nav-links a")];
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
