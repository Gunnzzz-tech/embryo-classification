# Deploy to Cloudflare Pages

Deploy the Embryo Fertility Admin app so others can use it at a public URL (no localhost needed).

---

## Option A: Deploy from your machine (CLI)

### 1. Install Wrangler (Cloudflare CLI)

```bash
npm install -g wrangler
```

Or use it without installing: `npx wrangler` (used below).

### 2. Log in to Cloudflare

```bash
npx wrangler login
```

A browser window opens; sign in or create a free Cloudflare account.

### 3. Create the Pages project (first time only)

```bash
npx wrangler pages project create embryo-fertility-admin
```

Choose your account and confirm. You can skip this if the project already exists in the dashboard.

### 4. Build and deploy

```bash
npm run deploy
```

This runs `npm run build` then deploys the `dist` folder to Cloudflare Pages.

### 5. Get your URL

After deploy, the CLI prints a URL like:

**https://embryo-fertility-admin.pages.dev**

Share this link; anyone can open the app (sign-in page first, then dashboard). Data stays in the browser (IndexedDB/localStorage) per user.

---

## Option B: Deploy via Git (GitHub/GitLab)

1. Push your code to **GitHub** or **GitLab**.
2. Go to [Cloudflare Dashboard](https://dash.cloudflare.com) → **Workers & Pages** → **Create** → **Pages** → **Connect to Git**.
3. Select the repo and configure:
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
   - **Root directory:** (leave default, or `/` if the app is at repo root)
4. Click **Save and Deploy**. Future pushes to the main branch will auto-deploy.

The `public/_redirects` file in this repo makes all routes (e.g. `/cases`, `/audit`) work correctly on the live URL.

---

## Notes

- **Free tier:** Cloudflare Pages is free for personal/small projects.
- **Data:** The app uses browser storage (IndexedDB, localStorage). Each user’s data stays in their browser; it is not stored on Cloudflare.
- **HTTPS:** Your Pages URL is served over HTTPS by default.
- **Custom domain:** In the Pages project → **Custom domains**, you can add your own domain.
