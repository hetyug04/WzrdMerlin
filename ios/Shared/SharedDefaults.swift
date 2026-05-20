import Foundation

// MARK: - Replace the App Group ID below with your registered identifier from the Apple Developer portal.
// Both the main app and the extension must share the same group ID in their entitlements.

enum SharedDefaults {
    static let appGroupID = "group.com.yourcompany.familytranslator"

    private static var store: UserDefaults {
        UserDefaults(suiteName: appGroupID) ?? .standard
    }

    static var selectedLanguage: Language {
        get {
            guard let raw = store.string(forKey: Keys.language),
                  let lang = Language(rawValue: raw) else { return .hindi }
            return lang
        }
        set { store.set(newValue.rawValue, forKey: Keys.language) }
    }

    static var apiKey: String {
        get { store.string(forKey: Keys.apiKey) ?? "" }
        set { store.set(newValue, forKey: Keys.apiKey) }
    }

    private enum Keys {
        static let language = "selectedLanguage"
        static let apiKey   = "claudeAPIKey"
    }
}
