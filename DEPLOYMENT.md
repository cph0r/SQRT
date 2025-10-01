# 🚀 Deployment Guide - Hugging Face Spaces

This guide walks you through deploying SQRT to Hugging Face Spaces with automated CI/CD using GitHub Actions.

## 📋 Prerequisites

1. **GitHub Account** - Your code repository
2. **Hugging Face Account** - Sign up at [huggingface.co](https://huggingface.co)
3. **Hugging Face Access Token** - For automated deployment

## 🔧 Setup Instructions

### Step 1: Create a Hugging Face Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure your Space:
   - **Space name**: `sqrt-selfie-rater` (or your preferred name)
   - **License**: MIT
   - **SDK**: Gradio
   - **Space hardware**: CPU Basic (free tier works fine)
   - **Visibility**: Public or Private
4. Click **"Create Space"**

### Step 2: Get Your Hugging Face Token

1. Go to [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Configure token:
   - **Name**: `SQRT_DEPLOY` (or any name you prefer)
   - **Role**: Write
4. Copy the generated token (you'll need it in the next step)

### Step 3: Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings → Secrets and variables → Actions**
3. Click **"New repository secret"**
4. Add the following secrets:

   **Secret 1:**
   - Name: `HF_TOKEN`
   - Value: Your Hugging Face token from Step 2

   **Secret 2:**
   - Name: `HF_SPACE_NAME`
   - Value: `YOUR_USERNAME/sqrt-selfie-rater` (e.g., `johndoe/sqrt-selfie-rater`)

### Step 4: Enable GitHub Actions

1. Go to your repository's **Actions** tab
2. If prompted, enable GitHub Actions for your repository
3. The workflow is already configured in `.github/workflows/deploy.yml`

### Step 5: Deploy! 🎉

**Automatic Deployment:**
- Every push to the `main` branch will automatically deploy to Hugging Face Spaces
- The GitHub Action will run and push your code to Hugging Face

**Manual Deployment:**
1. Go to **Actions** tab in your GitHub repository
2. Select **"Deploy to Hugging Face Spaces"** workflow
3. Click **"Run workflow"**
4. Select branch `main` and click **"Run workflow"**

## 🌐 Access Your Deployed App

After deployment completes (usually 1-2 minutes):
- Your app will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/sqrt-selfie-rater`
- Example: `https://huggingface.co/spaces/johndoe/sqrt-selfie-rater`

## 📊 Monitoring Deployment

### GitHub Actions
- Go to **Actions** tab to see deployment status
- Green checkmark ✅ = successful deployment
- Red X ❌ = deployment failed (check logs)

### Hugging Face Spaces
- Go to your Space page
- Check **"Logs"** tab to see build status
- Status indicator shows if app is running

## 🔄 Making Updates

1. Make changes to your code locally
2. Commit and push to `main` branch:
   ```bash
   git add .
   git commit -m "Your update message"
   git push origin main
   ```
3. GitHub Actions automatically deploys the update
4. Changes appear on Hugging Face within 1-2 minutes

## 🛠️ Troubleshooting

### Deployment fails with authentication error
- Check that `HF_TOKEN` secret is set correctly
- Ensure token has **Write** permissions
- Regenerate token if needed

### Build fails on Hugging Face
- Check Space logs on Hugging Face
- Verify all dependencies are in `requirements.txt`
- Ensure `app.py` is at repository root

### App crashes on startup
- Check Hugging Face Space logs
- Verify Python version compatibility
- Test locally before pushing: `python app.py`

### GitHub Action fails
- Check workflow logs in Actions tab
- Verify `HF_SPACE_NAME` format: `username/space-name`
- Ensure secrets are set in repository settings

## 📝 Configuration Files

### `.spacesconfig.yml`
Configures Hugging Face Space settings:
- SDK version
- App entry point
- Space metadata

### `.github/workflows/deploy.yml`
GitHub Actions workflow for automated deployment:
- Triggers on push to main
- Can be manually triggered
- Pushes code to Hugging Face

## 🎯 Best Practices

1. **Test Locally First**: Always test changes locally before pushing
2. **Small Commits**: Make incremental changes for easier debugging
3. **Monitor Logs**: Check both GitHub Actions and HF Space logs
4. **Use Branches**: Develop in feature branches, merge to main when ready
5. **Version Control**: Tag releases for easy rollback if needed

## 🔒 Security

- Never commit your `HF_TOKEN` to the repository
- Keep secrets in GitHub repository secrets
- Use minimal token permissions (Write only)
- Rotate tokens periodically

## 📚 Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## 💬 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review deployment logs
3. Open an issue on GitHub
4. Contact Hugging Face support

---

**Happy Deploying! 🚀**

