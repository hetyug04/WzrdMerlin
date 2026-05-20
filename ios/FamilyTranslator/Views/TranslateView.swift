import SwiftUI

struct TranslateView: View {
    @State private var inputText       = ""
    @State private var outputText      = ""
    @State private var isTranslating   = false
    @State private var errorMessage: String?
    @State private var selectedLanguage = SharedDefaults.selectedLanguage
    @State private var didCopy         = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    languagePicker
                    inputSection
                    translateButton
                    errorBanner
                    outputSection
                }
                .padding()
            }
            .navigationTitle("Family Translator")
            .onAppear { selectedLanguage = SharedDefaults.selectedLanguage }
        }
    }

    // MARK: - Subviews

    private var languagePicker: some View {
        HStack {
            Text("Translate to:").foregroundStyle(.secondary)
            Spacer()
            Picker("Language", selection: $selectedLanguage) {
                ForEach(Language.allCases) { lang in
                    Text("\(lang.emoji) \(lang.displayName)").tag(lang)
                }
            }
            .onChange(of: selectedLanguage) { _, new in
                SharedDefaults.selectedLanguage = new
                outputText  = ""
                errorMessage = nil
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("English", systemImage: "text.quote")
                .font(.caption).foregroundStyle(.secondary)

            ZStack(alignment: .topLeading) {
                TextEditor(text: $inputText)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 120)

                if inputText.isEmpty {
                    Text("Type or paste your message here…")
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 5).padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
            }
            .padding(8)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }

    private var translateButton: some View {
        Button(action: translate) {
            Group {
                if isTranslating {
                    HStack { ProgressView().tint(.white); Text("Translating…") }
                } else {
                    Label("Translate", systemImage: "arrow.down.circle.fill")
                }
            }
            .frame(maxWidth: .infinity).padding(.vertical, 4)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isTranslating)
    }

    @ViewBuilder
    private var errorBanner: some View {
        if let error = errorMessage {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                Text(error).font(.subheadline)
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.orange.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    @ViewBuilder
    private var outputSection: some View {
        if !outputText.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Label(selectedLanguage.displayName, systemImage: "character.bubble.fill")
                    .font(.caption).foregroundStyle(.secondary)

                Text(outputText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(Color(.secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 12))

                Button(action: copy) {
                    Label(
                        didCopy ? "Copied!" : "Copy to Clipboard",
                        systemImage: didCopy ? "checkmark.circle.fill" : "doc.on.doc.fill"
                    )
                    .frame(maxWidth: .infinity).padding(.vertical, 4)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .tint(didCopy ? .green : .accentColor)
            }
        }
    }

    // MARK: - Actions

    private func translate() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        isTranslating = true
        errorMessage  = nil
        outputText    = ""
        didCopy       = false
        let lang   = selectedLanguage
        let apiKey = SharedDefaults.apiKey
        Task {
            do {
                outputText = try await ClaudeService.translate(text: text, to: lang, apiKey: apiKey)
            } catch {
                errorMessage = error.localizedDescription
            }
            isTranslating = false
        }
    }

    private func copy() {
        UIPasteboard.general.string = outputText
        didCopy = true
        Task {
            try? await Task.sleep(for: .seconds(2))
            didCopy = false
        }
    }
}
