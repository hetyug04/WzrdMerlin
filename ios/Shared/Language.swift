import Foundation

enum Language: String, CaseIterable, Identifiable, Codable {
    case hindi      = "Hindi"
    case gujarati   = "Gujarati"
    case punjabi    = "Punjabi"
    case bengali    = "Bengali"
    case tamil      = "Tamil"
    case telugu     = "Telugu"
    case urdu       = "Urdu"
    case mandarin   = "Mandarin Chinese"
    case cantonese  = "Cantonese"
    case korean     = "Korean"
    case japanese   = "Japanese"
    case vietnamese = "Vietnamese"
    case tagalog    = "Tagalog"
    case spanish    = "Spanish"

    var id: String { rawValue }
    var displayName: String { rawValue }

    var emoji: String {
        switch self {
        case .hindi, .gujarati, .punjabi, .tamil, .telugu: return "🇮🇳"
        case .bengali:    return "🇧🇩"
        case .urdu:       return "🇵🇰"
        case .mandarin:   return "🇨🇳"
        case .cantonese:  return "🇭🇰"
        case .korean:     return "🇰🇷"
        case .japanese:   return "🇯🇵"
        case .vietnamese: return "🇻🇳"
        case .tagalog:    return "🇵🇭"
        case .spanish:    return "🇪🇸"
        }
    }
}
