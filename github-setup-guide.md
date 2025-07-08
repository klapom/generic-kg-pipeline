# GitHub Repository Setup - Schritt-für-Schritt Anleitung

## 1. Repository auf GitHub erstellen

### Schritt 1.1: Zur Repository-Erstellung navigieren
1. Öffne deinen Browser
2. Gehe zu: https://github.com/new
3. Stelle sicher, dass du bei GitHub eingeloggt bist

### Schritt 1.2: Repository-Einstellungen
Fülle das Formular wie folgt aus:

- **Repository name:** `generic-kg-pipeline`
- **Description:** `A flexible, plugin-based pipeline system for extracting knowledge graphs from documents`
- **Visibility:** 
  - ☑️ Private (empfohlen für Entwicklungsphase)
  - ⚪ Public (wenn es öffentlich sein soll)

### Schritt 1.3: Initialisierung (WICHTIG!)
**NICHT ankreuzen:**
- ❌ Add a README file
- ❌ Add .gitignore
- ❌ Choose a license

**Grund:** Wir haben diese Dateien bereits lokal erstellt.

### Schritt 1.4: Repository erstellen
- Klicke auf den grünen Button: **"Create repository"**

## 2. Repository mit lokalem Code verbinden

### Schritt 2.1: GitHub-Anweisungen kopieren
Nach der Repository-Erstellung zeigt GitHub eine Seite mit Anweisungen.
Unter "…or push an existing repository from the command line" siehst du:

```bash
git remote add origin https://github.com/YOUR_USERNAME/generic-kg-pipeline.git
git branch -M main
git push -u origin main
```

### Schritt 2.2: Befehle in Terminal ausführen
**Wichtig:** Ersetze `YOUR_USERNAME` mit deinem GitHub-Benutzernamen!

```bash
# Im generic-kg-pipeline Verzeichnis ausführen:
git remote add origin https://github.com/YOUR_USERNAME/generic-kg-pipeline.git
git push -u origin main
```

**Hinweis:** `git branch -M main` ist nicht nötig, da wir bereits den main Branch haben.

### Schritt 2.3: Authentifizierung
Beim ersten Push wirst du nach Anmeldedaten gefragt:
- **Username:** Dein GitHub-Username
- **Password:** Dein GitHub Personal Access Token (nicht dein Passwort!)

**Personal Access Token erstellen (falls nötig):**
1. Gehe zu: https://github.com/settings/tokens
2. Klicke "Generate new token (classic)"
3. Wähle Scope: `repo` (full control)
4. Kopiere den Token und verwende ihn als Passwort

## 3. GitHub Repository konfigurieren

### Schritt 3.1: Topics/Tags hinzufügen
1. Gehe zu deinem Repository: `https://github.com/YOUR_USERNAME/generic-kg-pipeline`
2. Klicke auf das ⚙️ Zahnrad neben "About"
3. Füge diese Topics hinzu:
   ```
   knowledge-graph
   llm
   document-processing
   pdf-parsing
   semantic-web
   python
   vllm
   triple-extraction
   rag
   ```
4. Klicke "Save changes"

### Schritt 3.2: Branch Protection einrichten
1. Gehe zu: Repository → Settings → Branches
2. Klicke "Add rule"
3. Branch name pattern: `main`
4. Aktiviere:
   - ☑️ Require a pull request before merging
   - ☑️ Require status checks to pass before merging
   - ☑️ Require branches to be up to date before merging
   - ☑️ Include administrators
5. Klicke "Create"

### Schritt 3.3: Issue Templates aktivieren
1. Gehe zu: Repository → Settings → Features
2. Unter "Issues" stelle sicher, dass es aktiviert ist
3. Die Templates werden automatisch aus `.github/ISSUE_TEMPLATE/` geladen

### Schritt 3.4: Wiki aktivieren (optional)
1. Gehe zu: Repository → Settings → Features
2. Aktiviere "Wikis"

## 4. Repository-Qualität verbessern

### Schritt 4.1: Repository-Banner erstellen
Füge diese Badges zu deiner README.md hinzu (nach dem Push):

```markdown
![Build Status](https://github.com/YOUR_USERNAME/generic-kg-pipeline/workflows/CI/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
```

### Schritt 4.2: GitHub Pages aktivieren (optional)
1. Gehe zu: Repository → Settings → Pages
2. Source: "Deploy from a branch"
3. Branch: `main`
4. Folder: `/ (root)` oder `/docs`

## Fehlerbehebung

### Problem: "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/generic-kg-pipeline.git
```

### Problem: Authentication failed
- Stelle sicher, dass du einen Personal Access Token verwendest
- Nicht dein GitHub-Passwort!

### Problem: Repository ist leer nach Push
- Überprüfe, ob du im richtigen Verzeichnis bist
- Führe `git status` aus, um Commits zu prüfen

## Erfolg verifizieren

Nach erfolgreichem Push solltest du sehen:
1. ✅ Alle Dateien sind auf GitHub sichtbar
2. ✅ GitHub Actions CI läuft automatisch
3. ✅ README.md wird schön angezeigt
4. ✅ Repository hat die konfigurierten Topics

## Nächste Schritte

Nach erfolgreichem Setup:
1. Erstelle Development Branch: `git checkout -b develop`
2. Lade Collaborators ein (falls Team-Projekt)
3. Konfiguriere GitHub Actions Secrets für CI/CD
4. Erstelle erste Issues für Feature-Entwicklung