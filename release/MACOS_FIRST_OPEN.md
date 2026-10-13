# Installing picasso-workflow on macOS

A short, click-by-click guide. No coding needed.

> **Why the extra step?** The installer isn't signed with a paid Apple
> certificate yet, so the first time you open it macOS asks you to confirm.
> This is a one-time step per download — the app itself is fine.

---

## 1. Which Macs are supported

The installer targets **Apple Silicon** Macs (M1 / M2 / M3 / M4) — check via the
Apple menu  → **About This Mac** → **Chip**. Download
`picasso-workflow-<version>-MacOSX-arm64.pkg`.

> **Intel Macs** aren't covered by the one-click installer. If About This Mac
> shows an Intel processor, install the classic way instead (create a conda env
> and `pip install picasso_workflow`), or ask the lab for help.

---

## 2. Open it the first time ("Open Anyway")

1. **Double-click** the `.pkg`. macOS shows *"…cannot be opened because it is
   from an unidentified developer"* or *"Apple could not verify…"*.
   Click **Done** — **do not** click "Move to Bin".
2. Open the Apple menu  → **System Settings** → **Privacy & Security**.
3. Scroll down. You'll see a line like
   *"picasso-workflow-….pkg was blocked to protect your Mac."*
   Click **Open Anyway**.
4. Confirm with Touch ID or your password if prompted.
5. **Double-click the `.pkg` again.** This time a dialog offers an **Open**
   button — click it, and the installer runs normally.

You only do steps 2–4 once per downloaded file.

---

## 3. Launch picasso-workflow

After installing, open it from **Spotlight** or **Launchpad**:

- **Spotlight:** press **⌘-Space**, type `picasso-workflow`, press Return.
  (Spotlight may take a minute to find it right after installing.)
- **Launchpad:** open Launchpad and click the picasso-workflow icon.

> The app is installed into your personal `~/Applications` folder (so it needs
> no admin rights and won't clash with other users). That folder isn't the same
> as Finder's "Applications" — that's why Spotlight/Launchpad are the easiest
> way to open it.

On first launch you'll see a **Getting Started** guide explaining where your
settings live and how to add your Confluence token.

---

## Alternative: the `.sh` installer (no security prompt, needs Terminal)

If you're comfortable with one Terminal command, the `.sh` installer skips the
security dialog entirely:

```sh
bash ~/Downloads/picasso-workflow-<version>-MacOSX-*.sh
```

Follow the prompts (press Return / type `yes` to accept the default location).

---

## For lab admins: distribute without any prompt

To hand colleagues an installer that opens with a plain double-click, remove the
"quarantine" flag **once** before putting it on a shared drive:

```sh
xattr -dr com.apple.quarantine /path/to/picasso-workflow-*-MacOSX-*.pkg
```

Then copy the cleaned `.pkg` to an **SMB/NAS share or a USB stick**. Colleagues
who open it from there won't see the security prompt.

> **Avoid OneDrive / Dropbox / iCloud for this** — cloud-sync clients often
> re-add the quarantine flag when the file downloads to another Mac, so the
> prompt can come back. A normal network share or USB stick is reliable.

---

*The one-time prompt goes away entirely once the installer is code-signed and
notarized with an Apple Developer ID — a paid option that can be added later.*
