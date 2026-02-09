# Deploy Financial Markets app to Render or Railway

The app runs as a Flask app. Use **Gunicorn** in production. Both Render and Railway set the `PORT` environment variable; the app already reads it.

---

## Prerequisites

- Code in a **Git** repo (GitHub, GitLab, or Bitbucket).
- **Note:** `heatmaps/` and `data/` are in `.gitignore`, so the deployed app won’t have pre-generated heatmaps or CSV data. The site will still run: landing page, News, Calendar (yfinance), and navigation work; Compare, Performance, and heatmap views need data (empty or run scripts in a build step if you add them later).

---

## Deploy to Render

1. **Sign up:** [render.com](https://render.com) → Sign up (GitHub login is easiest).

2. **New Web Service:** Dashboard → **New +** → **Web Service**.

3. **Connect repo:** Choose your **stock_analysis** (or parent) repo and connect. Select the repo that contains `app.py` and `requirements.txt`.

4. **Settings:**
   - **Name:** e.g. `financial-markets`
   - **Region:** Pick one close to you.
   - **Branch:** `main` (or your default).
   - **Root Directory:** If the app is in a subfolder (e.g. `stock_analysis`), set **Root Directory** to that folder. Otherwise leave blank.
   - **Runtime:** **Python 3**
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     gunicorn -w 2 -b 0.0.0.0:$PORT app:app
     ```
   - **Instance type:** Free (or paid if you prefer).

5. **Deploy:** Click **Create Web Service**. Render will build and start the app. The log will show the public URL (e.g. `https://financial-markets.onrender.com`).

6. **Use the URL:** Open that URL in the browser. Share it; no ngrok needed.

**Optional – environment variables:** In the service **Environment** tab you can add `PORT` (Render sets it automatically; only add if you need to override).

---

## Deploy to Railway

1. **Sign up:** [railway.app](https://railway.app) → **Login** (e.g. with GitHub).

2. **New project:** **New Project** → **Deploy from GitHub repo**. Select the repo that contains `app.py` (and set **Root Directory** to the app folder if it’s not the repo root).

3. **Configure service:** After the repo is linked, click the new service:
   - **Settings** → **Build:**  
     - **Builder:** Nixpacks (default) or **Dockerfile** if you add one.  
     - For Nixpacks, Railway usually detects Python. Ensure **Build Command** runs:
       ```bash
       pip install -r requirements.txt
       ```
   - **Settings** → **Deploy** / **Start Command** (or in **Variables** as `RAILWAY_RUN_COMMAND` or in a **Procfile**):  
     ```bash
     gunicorn -w 2 -b 0.0.0.0:$PORT app:app
     ```
   - Railway sets `PORT` automatically; the app already uses it.

4. **Public URL:** In **Settings** → **Networking** (or **Deployments**), generate a **Public Domain** for the service. You’ll get a URL like `https://your-app.up.railway.app`.

5. **Deploy:** Push to the connected branch or trigger a deploy. Use the public URL to open and share the app.

**Optional – Procfile:** In the app root (or Root Directory) you can add a **Procfile** so Railway knows how to start the app:

```
web: gunicorn -w 2 -b 0.0.0.0:$PORT app:app
```

---

## Summary

| Step        | Render | Railway |
|------------|--------|---------|
| Build      | `pip install -r requirements.txt` | Same (or Nixpacks default) |
| Start      | `gunicorn -w 2 -b 0.0.0.0:$PORT app:app` | Same (or Procfile) |
| PORT       | Set by Render | Set by Railway |
| Public URL | e.g. `https://your-app.onrender.com` | e.g. `https://your-app.up.railway.app` |

After deployment, use the HTTPS URL from the dashboard; no tunnel or “connection not secure” warning for your own domain. If you later add a build step that generates `heatmaps/` or `data/`, those features will work on the deployed app as well.
