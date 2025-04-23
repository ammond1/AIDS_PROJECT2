
# AIDS_PROJECT2

AIDS_PROJECT2 scrapes Reddit posts related to AIDS, summarizes them using OpenAI's GPT API, stores the summaries in Firebase Realtime Database, and provides a web interface hosted on Firebase to view the results.

---

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ammond1/AIDS_PROJECT2.git
   cd AIDS_PROJECT2
   ```

2. Set up a Python virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Environment Variables

Create a `.env` file in the root directory with the following contents:

```
REDDIT_ID=your_reddit_client_id
REDDIT_SECRET=your_reddit_client_secret
OPENAI_API_KEY=your_openai_api_key
FIREBASE_DB_URL=your_firebase_realtime_database_url
FIREBASE_CREDENTIALS_PATH=path_to_your_firebase_service_account.json
```

You need a Firebase Realtime Database set up and a service account JSON credentials file.

---

## 🖥️ Setting up Firebase Hosting and Realtime Database

1. Install Firebase CLI:

   ```bash
   npm install -g firebase-tools
   ```

2. Login to Firebase:

   ```bash
   firebase login
   ```

3. Initialize Firebase in the project directory:

   ```bash
   firebase init
   ```

   - Select **Hosting** and **Database**.
   - Set `public/` as your hosting directory.
   - Configure as a single-page app if needed (yes).
   - Set up your Firebase Realtime Database rules and URL.

4. Set up `public/` frontend to read data from your Realtime Database (make sure your `index.html` includes Firebase SDK and database references).

Example snippet in your `index.html`:

```html
<script src="https://www.gstatic.com/firebasejs/9.0.0/firebase-app.js"></script>
<script src="https://www.gstatic.com/firebasejs/9.0.0/firebase-database.js"></script>
<script>
  const firebaseConfig = {
    apiKey: "YOUR_API_KEY",
    authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
    databaseURL: "YOUR_DATABASE_URL",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT_ID.appspot.com",
    messagingSenderId: "YOUR_SENDER_ID",
    appId: "YOUR_APP_ID"
  };

  const app = firebase.initializeApp(firebaseConfig);
  const db = firebase.database();

  db.ref('summaries/').on('value', (snapshot) => {
    const data = snapshot.val();
    // Display data on your page
  });
</script>
```

---

## 🧠 Running the Backend

Run the Python script to scrape and summarize Reddit posts and upload them to Firebase:

```bash
python main.py
```

This script will:
- Connect to Reddit
- Fetch posts based on keywords
- Summarize the posts using OpenAI
- Upload the results to your Firebase Realtime Database under `summaries/`

Make sure your service account JSON and database URL are correctly referenced.

---

## 🌐 Hosting the Frontend

Deploy the frontend to Firebase Hosting:

```bash
firebase deploy
```

After deploying, you can access the app at the provided Firebase Hosting URL.

---

## ✅ Requirements

- Python 3.8+
- PRAW
- OpenAI SDK
- Flask (optional for local development)
- python-dotenv
- firebase-admin (Python SDK)
- Firebase CLI

Install Firebase Admin SDK:

```bash
pip install firebase-admin
```

---

## 📌 Notes

- Firebase hosting and realtime database configuration files are included.
- Use responsibly and comply with Reddit’s, OpenAI’s, and Firebase's usage policies.
