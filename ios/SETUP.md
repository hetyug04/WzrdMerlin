# Family Translator — Setup Guide

## Prerequisites
- Xcode 15+
- [XcodeGen](https://github.com/yonaskolb/XcodeGen): `brew install xcodegen`
- An Apple Developer account (free tier works for device testing)
- A Claude API key from [console.anthropic.com](https://console.anthropic.com)

---

## Step 1 — Register an App Group

App Groups let the main app and the Share Extension share UserDefaults (API key, language preference).

1. Go to [developer.apple.com](https://developer.apple.com) → **Certificates, IDs & Profiles** → **App Groups**
2. Create a new group, e.g. `group.com.yourcompany.familytranslator`
3. Enable this group on **two** App IDs: your main app and the extension

Then replace `group.com.yourcompany.familytranslator` in **three places**:
- `Shared/SharedDefaults.swift` — `appGroupID`
- `FamilyTranslator/FamilyTranslator.entitlements`
- `TranslateExtension/TranslateExtension.entitlements`

Also replace `com.yourcompany` in `project.yml` with your real bundle ID prefix.

---

## Step 2 — Generate the Xcode project

```bash
cd ios/
xcodegen generate
open FamilyTranslator.xcodeproj
```

---

## Step 3 — Configure signing in Xcode

1. Select the **FamilyTranslator** target → **Signing & Capabilities**
   - Set your Team
   - Enable **App Groups** capability → add your group ID
2. Select the **TranslateExtension** target → **Signing & Capabilities**
   - Set the same Team
   - Enable **App Groups** capability → add the same group ID

---

## Step 4 — Run & set your API key

1. Build and run on a device (Share Extensions don't work in Simulator)
2. Open the **Settings** tab → paste your Claude API key → **Save**
3. Open any app (Messages, Notes, Safari…), select some English text
4. Tap **Share** → **Translate for Family**
5. The extension auto-translates and shows a **Copy & Return** button

---

## File structure

```
ios/
├── project.yml                        ← XcodeGen spec
├── Shared/                            ← compiled into BOTH targets
│   ├── Language.swift                 ← enum of 14 languages
│   ├── SharedDefaults.swift           ← App Group UserDefaults wrapper
│   └── ClaudeService.swift            ← URLSession → Claude API
├── FamilyTranslator/                  ← main app target
│   ├── FamilyTranslatorApp.swift
│   ├── ContentView.swift              ← TabView (Translate + Settings)
│   ├── Views/
│   │   ├── TranslateView.swift        ← standalone translate screen
│   │   └── SettingsView.swift         ← API key + default language
│   ├── FamilyTranslator.entitlements
│   └── Info.plist
└── TranslateExtension/                ← Share Extension target
    ├── ShareViewController.swift      ← UIViewController entry point
    ├── ShareTranslateView.swift       ← SwiftUI translate UI
    ├── TranslateExtension.entitlements
    └── Info.plist
```

---

## How the Share Extension flow works

```
User selects text in any app
        │
        ▼
Tap Share sheet → "Translate for Family"
        │
        ▼
ShareViewController.viewDidLoad()
  └─ NSItemProvider.loadItem(UTType.plainText)
        │
        ▼
ShareTranslateView appears
  └─ .task { runTranslation() }   ← auto-starts immediately
        │
        ▼
ClaudeService.translate()
  └─ POST https://api.anthropic.com/v1/messages
        │
        ▼
Translation appears
  └─ "Copy & Return" button
        │
        ▼
UIPasteboard.general.string = translation
extensionContext?.completeRequest(...)
        │
        ▼
User is back in the original app → paste & send
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "No API key set" in extension | Save the key in the main app first; both share the same App Group |
| Extension not appearing in share sheet | Make sure `NSExtensionActivationSupportsText = true` in the extension's Info.plist |
| Shared UserDefaults returns nil | App Group ID mismatch between entitlements and `SharedDefaults.appGroupID` |
| Build error: module not found | Run `xcodegen generate` again after changing `project.yml` |
| Translation times out | Check network; the extension has a 30 s URLSession timeout |
