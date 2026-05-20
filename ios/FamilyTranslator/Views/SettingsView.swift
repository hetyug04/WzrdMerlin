import SwiftUI

struct SettingsView: View {
    @State private var apiKey           = SharedDefaults.apiKey
    @State private var selectedLanguage = SharedDefaults.selectedLanguage
    @State private var showKey          = false
    @State private var saved            = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Default Language", selection: $selectedLanguage) {
                        ForEach(Language.allCases) { lang in
                            Text("\(lang.emoji) \(lang.displayName)").tag(lang)
                        }
                    }
                } header: {
                    Text("Language")
                } footer: {
                    Text("Pre-selected whenever you open the Share Extension.")
                }

                Section {
                    HStack {
                        Group {
                            if showKey {
                                TextField("sk-ant-api03-…", text: $apiKey)
                                    .autocorrectionDisabled()
                                    .textInputAutocapitalization(.never)
                            } else {
                                SecureField("sk-ant-api03-…", text: $apiKey)
                            }
                        }
                        .font(.system(.body, design: .monospaced))

                        Button { showKey.toggle() } label: {
                            Image(systemName: showKey ? "eye.slash" : "eye")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.borderless)
                    }
                } header: {
                    Text("Claude API Key")
                } footer: {
                    Text("Get your key at console.anthropic.com. Stored only on this device, shared with the extension via App Group.")
                }

                Section {
                    Button(action: save) {
                        HStack {
                            Text(saved ? "Saved!" : "Save Settings")
                                .frame(maxWidth: .infinity)
                            if saved {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                            }
                        }
                    }
                    .disabled(saved)
                }
            }
            .navigationTitle("Settings")
        }
    }

    private func save() {
        SharedDefaults.selectedLanguage = selectedLanguage
        SharedDefaults.apiKey = apiKey.trimmingCharacters(in: .whitespaces)
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil, from: nil, for: nil
        )
        saved = true
        Task {
            try? await Task.sleep(for: .seconds(2))
            saved = false
        }
    }
}
