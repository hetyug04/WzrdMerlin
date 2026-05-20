import Foundation

enum ClaudeService {
    private static let endpoint = URL(string: "https://api.anthropic.com/v1/messages")!
    private static let model    = "claude-sonnet-4-20250514"

    static func translate(text: String, to language: Language, apiKey: String) async throws -> String {
        let key = apiKey.trimmingCharacters(in: .whitespaces)
        guard !key.isEmpty else { throw TranslationError.missingAPIKey }

        var req = URLRequest(url: endpoint)
        req.httpMethod = "POST"
        req.timeoutInterval = 30
        req.setValue("application/json",  forHTTPHeaderField: "Content-Type")
        req.setValue(key,                 forHTTPHeaderField: "x-api-key")
        req.setValue("2023-06-01",        forHTTPHeaderField: "anthropic-version")

        let system = """
            Translate the following English text to \(language.displayName). \
            Use casual warm conversational tone as if speaking to a parent or grandparent. \
            Avoid formal or literary phrasing. Keep names as-is.
            """

        let body: [String: Any] = [
            "model":      model,
            "max_tokens": 1024,
            "system":     system,
            "messages":   [["role": "user", "content": text]]
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: req)

        guard let http = response as? HTTPURLResponse else {
            throw TranslationError.networkError("No HTTP response")
        }

        switch http.statusCode {
        case 200:
            let decoded = try JSONDecoder().decode(ClaudeResponse.self, from: data)
            guard let output = decoded.content.first?.text, !output.isEmpty else {
                throw TranslationError.emptyResponse
            }
            return output
        case 401:
            throw TranslationError.invalidAPIKey
        case 429:
            throw TranslationError.rateLimited
        default:
            throw TranslationError.serverError(http.statusCode)
        }
    }
}

// MARK: - Errors

enum TranslationError: LocalizedError {
    case missingAPIKey
    case invalidAPIKey
    case rateLimited
    case networkError(String)
    case serverError(Int)
    case emptyResponse

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "No API key set. Open the main app → Settings → add your Claude API key."
        case .invalidAPIKey:
            return "Invalid API key. Check your key in Settings."
        case .rateLimited:
            return "Too many requests. Please wait a moment and try again."
        case .networkError(let msg):
            return "Network error: \(msg)"
        case .serverError(let code):
            return "Server error (\(code)). Please try again."
        case .emptyResponse:
            return "No translation returned. Please try again."
        }
    }
}

// MARK: - Response model

private struct ClaudeResponse: Decodable {
    let content: [Block]
    struct Block: Decodable { let text: String }
}
