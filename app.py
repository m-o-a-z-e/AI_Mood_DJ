import os
import random
import math
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import qrcode
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
app.config['SECRET_KEY'] = os.urandom(24) 

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_PERMANENT'] = False

# Spotify Credentials
CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['Surprise', 'Disgust', 'Angry', 'Happy', 'Neutral', 'Sad', 'Neutral']

# Database & Login Setup
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Global State ---
# Stores the latest detected mood and confidence for the active session
current_mood_state = {"mood": "Neutral", "confidence": 0.0}
global_frame = None 
current_playlist_url = None 

# --- Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- AI Model Architecture ---
class CustomDeepEmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomDeepEmotionNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Spotify Integration Logic ---
class SpotifyPlaylistMaker:
    def __init__(self, client_id, client_secret):
        scope = "playlist-modify-public user-library-read"
        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=client_id, client_secret=client_secret,
                redirect_uri="http://127.0.0.1:8888/callback", scope=scope
            )
        )
        try:
            self.user_id = self.sp.current_user()['id']
            print(f"Spotify Connected: {self.user_id}")
        except Exception as e: 
            print(f"Spotify Connection Failed: {e}")

        # Mood mapping configuration
        self.mood_config = {
            'Happy': { 'queries': {'en': ['Happy Pop Hits', 'Summer Vibes'], 'ar': ['Amr Diab Upbeat', 'Hassan Shakosh']}, 'features': {'min_valence': 0.6, 'min_energy': 0.65} },
            'Sad': { 'queries': {'en': ['Sad Piano', 'Adele'], 'ar': ['Tamer Ashour Sad', 'Muslim']}, 'features': {'max_valence': 0.4, 'max_energy': 0.55} },
            'Angry': { 'queries': {'en': ['Heavy Metal', 'Eminem'], 'ar': ['Marwan Pablo', 'Wegz']}, 'features': {'min_energy': 0.8} },
            'Neutral': { 'queries': {'en': ['Lo-Fi', 'Coldplay'], 'ar': ['Cairokee', 'Hamza Namira']}, 'features': {'max_energy': 0.7, 'min_valence': 0.3} },
            'Surprise': { 'queries': {'en': ['EDM Drops', 'Skrillex'], 'ar': ['Ahmed Saad', 'Reda El Bahrawy']}, 'features': {'min_energy': 0.8, 'min_danceability': 0.7} },
            'Fear': { 'queries': {'en': ['Horror Soundtracks'], 'ar': ['Dark Oud']}, 'features': {'max_valence': 0.2} },
            'Disgust': { 'queries': {'en': ['Grunge', 'Nirvana'], 'ar': ['Underground Rap']}, 'features': {'max_valence': 0.35} }
        }

    def filter_tracks_batch(self, track_ids, filters):
        """Filters tracks based on audio features."""
        valid_uris = []
        try:
            for i in range(0, len(track_ids), 50):
                batch = track_ids[i:i+50]
                features_list = self.sp.audio_features(batch)
                for idx, features in enumerate(features_list):
                    if not features: continue
                    is_valid = True
                    if 'min_valence' in filters and features['valence'] < filters['min_valence']: is_valid = False
                    if 'max_valence' in filters and features['valence'] > filters['max_valence']: is_valid = False
                    if 'min_energy' in filters and features['energy'] < filters['min_energy']: is_valid = False
                    if 'max_energy' in filters and features['energy'] > filters['max_energy']: is_valid = False
                    if 'min_danceability' in filters and features['danceability'] < filters['min_danceability']: is_valid = False
                    if is_valid: valid_uris.append(f"spotify:track:{batch[idx]}")
        except: return [f"spotify:track:{tid}" for tid in track_ids]
        return valid_uris

    def create_playlist(self, mood, lang='mix', therapy_mode='match'):
        mood = mood.capitalize()
        original_mood = mood 
        
        # Apply Iso-Principle logic if therapy mode is active
        if therapy_mode == 'cheer':
            if mood == 'Sad': mood = 'Happy'
            elif mood == 'Angry': mood = 'Neutral'
            elif mood == 'Fear': mood = 'Happy'
            elif mood == 'Disgust': mood = 'Happy'
        
        if mood not in self.mood_config: mood = 'Neutral'
        config = self.mood_config[mood]
        filters = config.get('features', {})
        
        # Select query seeds based on language
        selected_queries = []
        if lang == 'mix':
            selected_queries.extend(random.sample(config['queries']['en'], 2))
            selected_queries.extend(random.sample(config['queries']['ar'], 2))
        elif lang in ['ar', 'en']:
            pool = config['queries'][lang]
            selected_queries = random.sample(pool, min(4, len(pool)))
        else: selected_queries = config['queries']['en'][:3]

        print(f"Generating {mood} Playlist (Strategy: {therapy_mode})")
        try:
            strategy_text = "Therapy Mix" if therapy_mode == 'cheer' else "Vibe Match"
            playlist_name = f"AI DJ: {original_mood} -> {mood} ({lang.upper()})"
            playlist = self.sp.user_playlist_create(user=self.user_id, name=playlist_name, public=True, description=f"Generated by AI Mood DJ.")
            
            candidates_ids = []
            for query in selected_queries:
                results = self.sp.search(q=query, limit=15, type='track', market='EG')
                for item in results['tracks']['items']: candidates_ids.append(item['id'])
            
            candidates_ids = list(set(candidates_ids))
            random.shuffle(candidates_ids)
            final_tracks_uris = self.filter_tracks_batch(candidates_ids, filters)
            
            # Fallback if filtering is too strict
            if not final_tracks_uris: final_tracks_uris = [f"spotify:track:{tid}" for tid in candidates_ids[:20]]
            final_selection = final_tracks_uris[:30]

            if final_selection:
                self.sp.playlist_add_items(playlist['id'], final_selection)
                return playlist['external_urls']['spotify']
            else: return None
        except Exception as e:
            print(f"Error: {e}")
            return None

# --- Initialization ---
model = CustomDeepEmotionNet(num_classes=7).to(device)
try:
    model.load_state_dict(torch.load('custom_emotion_model.pth', map_location=device))
    model.eval()
    print("Model Loaded")
except: print("Model Failed")

dj = SpotifyPlaylistMaker(CLIENT_ID, CLIENT_SECRET)
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Video Streaming Logic ---
def generate_frames():
    global global_frame
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        
        # Mirror effect
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        detected_mood = "Scanning..."
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            try:
                # Preprocess face for the model
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0: continue
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(roi_rgb)
                tensor = transform(pil_img).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    out = model(tensor)
                    probs = F.softmax(out, dim=1).cpu().numpy().squeeze()
                
                idx = np.argmax(probs)
                detected_mood = class_names[idx]
                confidence = probs[idx] * 100
                
                # Update global state
                current_mood_state["mood"] = detected_mood
                current_mood_state["confidence"] = float(confidence)
                
                cv2.putText(frame, f"{detected_mood} ({confidence:.1f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except: pass
            
        # Store clean frame copy for snapshots if needed, here we store annotated
        global_frame = frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- Graphic Helper Functions (Pillow) ---

def create_glow(width, height, color, radius=30):
    """Creates a neon glow layer using Gaussian Blur."""
    glow = Image.new('RGBA', (width + radius*4, height + radius*4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    draw.rounded_rectangle((radius, radius, width+radius*3, height+radius*3), radius=40, fill=color)
    return glow.filter(ImageFilter.GaussianBlur(radius))

def draw_neon_text(img, text, x, y, font, text_color, glow_color, glow_radius=15):
    """Draws text with a glowing neon effect."""
    draw = ImageDraw.Draw(img)
    
    # 1. Create Glow Layer
    txt_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
    txt_draw = ImageDraw.Draw(txt_layer)
    
    # Draw text multiple times to intensify glow
    for offset in range(1, 4):
        txt_draw.text((x, y), text, font=font, fill=glow_color, anchor="mm")
    
    # Apply Blur
    txt_layer = txt_layer.filter(ImageFilter.GaussianBlur(glow_radius))
    
    # 2. Composite Glow with Original Image
    img.alpha_composite(txt_layer)
    
    # 3. Draw Sharp Text on top
    draw.text((x, y), text, font=font, fill=text_color, anchor="mm")

def draw_digital_wave(img, x_center, y_center, width, height, color1, color2):
    """Draws a digital equalizer visualizer."""
    draw = ImageDraw.Draw(img)
    num_bars = 25
    bar_width = 12
    gap = 8
    total_wave_width = (num_bars * bar_width) + ((num_bars - 1) * gap)
    start_x = x_center - (total_wave_width // 2)
    
    for i in range(num_bars):
        # Calculate bell curve shape for the wave
        dist_from_center = abs(i - (num_bars // 2))
        bar_height = int(height * (1 - (dist_from_center / (num_bars/1.5)))) 
        
        # Add randomness for dynamic effect
        bar_height += random.randint(-10, 10)
        if bar_height < 10: bar_height = 10
        
        current_x = start_x + (i * (bar_width + gap))
        
        # Alternate colors
        fill_color = color1 if i % 2 == 0 else color2
        
        draw.rounded_rectangle(
            (current_x, y_center - bar_height//2, current_x + bar_width, y_center + bar_height//2),
            radius=5, fill=fill_color
        )

def add_corners(im, rad):
    """Crops an image to have rounded corners."""
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', im.size, 255)
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im

# --- Main Graphic Generator Function ---

def create_story_image(frame, mood, playlist_url):
    """Generates the final 9:16 Instagram Story image."""
    NEON_CYAN = '#00ffcc'
    NEON_PURPLE = '#bd00ff'
    BG_DARK = '#0a0a12'
    
    W, H = 1080, 1920
    
    # 1. Background Layer
    base = Image.new('RGBA', (W, H), BG_DARK)
    
    # 2. Load Fonts (Fallback to default if custom fonts missing)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 100) 
        mood_font = ImageFont.truetype("arialbd.ttf", 65)
        small_font = ImageFont.truetype("arial.ttf", 45)
    except:
        title_font = ImageFont.load_default()
        mood_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # 3. Draw Neon Title
    draw_neon_text(base, "AI MOOD DJ", W//2, 200, title_font, 'white', NEON_CYAN, glow_radius=30)

    # 4. Process Camera Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cam_img = Image.fromarray(frame_rgb).convert("RGBA")
    
    # Resize and Crop
    target_w = 900
    ratio = target_w / cam_img.width
    target_h = int(cam_img.height * ratio)
    cam_img = cam_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    cam_img = add_corners(cam_img, 60)
    
    # 5. Add Neon Border
    glow_layer = create_glow(target_w, target_h, NEON_CYAN, radius=40)
    
    # Center Image
    img_y = 450
    glow_x = (W - glow_layer.width) // 2
    glow_y = img_y - 40 
    base.alpha_composite(glow_layer, (glow_x, glow_y))
    base.paste(cam_img, ((W - target_w)//2, img_y), cam_img)
    
    # Draw sharp outline
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle(
        ((W - target_w)//2, img_y, (W + target_w)//2, img_y + target_h),
        radius=60, outline=NEON_CYAN, width=5
    )

    # 6. Draw Mood Text
    text_y = img_y + target_h + 100
    draw_neon_text(base, f"MOOD DETECTED: {mood.upper()}", W//2, text_y, mood_font, NEON_CYAN, NEON_PURPLE, glow_radius=20)
    
    # 7. Draw Sound Wave Visualization
    draw_digital_wave(base, W//2, text_y + 100, 800, 100, NEON_CYAN, NEON_PURPLE)

    # 8. Generate QR Code
    if playlist_url:
        qr = qrcode.QRCode(box_size=14, border=1)
        qr.add_data(playlist_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")
        qr_img = qr_img.resize((350, 350))
        
        # QR Background Glow
        qr_y = 1350
        qr_bg_size = 400
        qr_glow = create_glow(qr_bg_size, qr_bg_size, NEON_PURPLE, radius=40)
        
        base.alpha_composite(qr_glow, ((W - qr_glow.width)//2, qr_y - 40))
        
        # White box behind QR
        draw.rounded_rectangle(
            ((W - qr_bg_size)//2, qr_y, (W + qr_bg_size)//2, qr_y + qr_bg_size),
            radius=30, fill='white'
        )
        
        base.paste(qr_img, ((W - 350)//2, qr_y + 25), qr_img)
        
        # Call to Action Text
        scan_y = qr_y + qr_bg_size + 80
        draw_neon_text(base, "SCAN TO LISTEN 🎧", W//2, scan_y, small_font, 'white', NEON_PURPLE, glow_radius=15)

    return base.convert("RGB")

# --- Routes ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/experience')
@login_required
def experience(): return render_template('experience.html', name=current_user.username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('experience'))
        else: flash('Login Failed.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if User.query.filter_by(email=request.form.get('email')).first(): flash('Exists', 'error')
        else:
            new = User(username=request.form.get('username'), email=request.form.get('email'), password=generate_password_hash(request.form.get('password'), method='pbkdf2:sha256'))
            db.session.add(new); db.session.commit(); login_user(new)
            return redirect(url_for('experience'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout(): logout_user(); return redirect(url_for('index'))

@app.route('/video_feed')
@login_required
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/create_playlist_api', methods=['POST'])
@login_required
def create_playlist_api():
    global current_playlist_url 
    mood = current_mood_state["mood"]
    data = request.get_json()
    lang = data.get('language', 'mix'); therapy = data.get('therapy_mode', 'match')
    
    url = dj.create_playlist(mood, lang=lang, therapy_mode=therapy)
    if url:
        current_playlist_url = url 
        return jsonify({"status": "success", "url": url, "mood": mood, "lang": lang})
    else: return jsonify({"status": "error"})

@app.route('/snapshot')
@login_required
def snapshot():
    global global_frame, current_mood_state, current_playlist_url
    if global_frame is not None:
        # Generate Story Image
        story_image = create_story_image(global_frame, current_mood_state['mood'], current_playlist_url)
        
        # Convert to bytes for response
        img_io = BytesIO()
        story_image.save(img_io, 'JPEG', quality=90)
        img_io.seek(0)
        return Response(img_io, mimetype='image/jpeg')
    else: return "Camera not active", 404

if __name__ == "__main__":
    print("Server Running...")
    app.run(debug=True, port=5000)