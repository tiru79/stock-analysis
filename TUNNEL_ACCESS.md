# Accessing the app via tunnel (ngrok / localtunnel)

If **http://127.0.0.1:5000** works on your Mac but the **ngrok** URL shows **"This site can't provide a secure connection"** or **ERR_SSL_PROTOCOL_ERROR** ("sent an invalid response"), your Mac’s connection to ngrok’s domain is failing TLS. The tunnel and app are fine; the problem is only this machine’s outbound TLS to ngrok.

## Option A: Try Chrome on this Mac

Some networks or macOS TLS settings break Safari but not Chrome (Chrome uses a different TLS stack). Install [Google Chrome](https://www.google.com/chrome/) and open the ngrok **https** URL in Chrome. If it loads, you can use the app from your Mac in Chrome.

## Option B: Open the tunnel URL from another device (most reliable)

Your Mac’s connection to the tunnel domain (ngrok-free.dev) may fail TLS; other devices (phone, another PC) usually work.

1. **Keep the app and tunnel running on your Mac:**
   - Terminal 1: `./run_webapp.sh` (Flask)
   - Terminal 2: **ngrok** → `ngrok http 127.0.0.1:5000`  
     **or** **localtunnel** → `npx localtunnel --port 5000`

2. **Get the public URL:**
   - From the ngrok terminal output, or
   - Open **http://127.0.0.1:4040** in your Mac’s browser (ngrok dashboard) and copy the “Forwarding” https URL.

3. **On your phone or another computer**, open that **https** URL in the browser:
   - **Ngrok:** Click **“Visit Site”** on the interstitial if shown.
   - **Localtunnel:** Enter your IP or click through if asked.
   - If it fails on Wi‑Fi, try on **cellular/mobile data** (in case your network blocks or alters TLS).

4. **Share that same URL** with others.

## Copy the public URL from your Mac

- **Ngrok:** In the terminal where ngrok is running, copy the line like  
  `Forwarding   https://something.ngrok-free.dev -> http://127.0.0.1:5000`  
  Or open **http://127.0.0.1:4040** in your browser (ngrok’s local dashboard) and copy the https URL from there, then type or paste it on your phone.

## Quick reference

| Step | Command / URL |
|------|----------------|
| Start app | `./run_webapp.sh` |
| Start ngrok | `ngrok http 127.0.0.1:5000` |
| Start localtunnel | `npx localtunnel --port 5000` |
| Local URL | http://127.0.0.1:5000 |
| Ngrok dashboard (copy URL) | http://127.0.0.1:4040 |
| Public URL | Use the **https** URL from ngrok or localtunnel (open from phone/other device) |
