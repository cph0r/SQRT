# 🚀 Quick Deploy to Hugging Face - 5 Minutes Setup

## ⚡ TL;DR - Fast Track

1. **Create Hugging Face Space**: [huggingface.co/new-space](https://huggingface.co/new-space)
2. **Get HF Token**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **Add GitHub Secrets**: `HF_TOKEN` + `HF_SPACE_NAME`
4. **Push to main**: Automatic deployment! 🎉

---

## 📝 Step-by-Step (5 minutes)

### 1️⃣ Create Hugging Face Space (1 min)

Visit: https://huggingface.co/new-space

Fill in:
- **Name**: `sqrt-selfie-rater`
- **License**: MIT
- **SDK**: Gradio
- **Hardware**: CPU Basic (free)

Click **"Create Space"**

### 2️⃣ Get Your Token (1 min)

Visit: https://huggingface.co/settings/tokens

Click **"New token"**:
- **Name**: `SQRT_DEPLOY`
- **Role**: **Write** ✅

Copy the token 📋

### 3️⃣ Configure GitHub (2 min)

Go to: Your GitHub repo → Settings → Secrets and variables → Actions

Add **Secret #1**:
```
Name:  HF_TOKEN
Value: [paste your Hugging Face token]
```

Add **Secret #2**:
```
Name:  HF_SPACE_NAME
Value: YOUR_USERNAME/sqrt-selfie-rater
```
(Replace `YOUR_USERNAME` with your actual HF username)

### 4️⃣ Deploy! (1 min)

**Option A - Automatic (on every push):**
```bash
git add .
git commit -m "Deploy to Hugging Face"
git push origin main
```

**Option B - Manual trigger:**
1. Go to: GitHub repo → Actions tab
2. Click "Deploy to Hugging Face Spaces"
3. Click "Run workflow" → "Run workflow"

### 5️⃣ Done! 🎉

Your app will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/sqrt-selfie-rater
```

Check deployment status:
- **GitHub**: Actions tab (workflow progress)
- **Hugging Face**: Your Space → Logs tab

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| ❌ "Authentication failed" | Regenerate HF token with **Write** permissions |
| ❌ "Space not found" | Check `HF_SPACE_NAME` format: `username/space-name` |
| ❌ Build fails | Check Space logs on HuggingFace for details |
| ⏳ Takes too long | First build ~2-3 min (installs dependencies) |

---

## 📊 What Happens When You Deploy?

1. **GitHub Action triggers** on push to `main`
2. **Code is pushed** to Hugging Face
3. **HF builds** the Space (installs requirements)
4. **App launches** automatically
5. **Live in 2-3 minutes!** ⚡

---

## 🔄 Making Updates

Just push to main:
```bash
git add .
git commit -m "Update feature X"
git push origin main
```

Auto-deploys in ~1-2 minutes! 🚀

---

## 💡 Tips

- ✅ Test locally first: `python app.py`
- ✅ Check logs if issues occur
- ✅ Free tier works great for this app
- ✅ Upgrade to GPU for faster face detection (optional)

---

**Need more details?** See [DEPLOYMENT.md](DEPLOYMENT.md) for complete guide.

