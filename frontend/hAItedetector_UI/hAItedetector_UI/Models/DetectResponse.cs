using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace hAItedetector_UI.Models
{
    public class DetectResponse
    {
        // Scoruri sigmoid per etichetă (0.0 – 1.0)
        // ex: { "spam": 0.9987, "hate_speech": 0.0098 }
        [JsonPropertyName("scores")]
        public Dictionary<string, double> Scores { get; set; } = new();

        // Decizii binare per etichetă (true dacă scorul ≥ prag)
        // ex: { "spam": true, "hate_speech": false }
        [JsonPropertyName("decisions")]
        public Dictionary<string, bool> Decisions { get; set; } = new();

        // Limba detectată automat sau forțată din UI
        [JsonPropertyName("detected_lang")]
        public string DetectedLang { get; set; } = "en";

        // Subtipul de spam — null dacă spam nu e detectat
        [JsonPropertyName("spam_subtype")]
        public string? SpamSubtype { get; set; }

        // Ținta hate speech — null dacă hate_speech nu e detectat
        [JsonPropertyName("hate_target")]
        public string? HateTarget { get; set; }

        // Severitatea hate speech: "low" / "medium" / "high" / null
        [JsonPropertyName("hate_severity")]
        public string? HateSeverity { get; set; }

        // Textul cenzurat — null dacă censor_hate=false sau hate nedetectat
        [JsonPropertyName("censored_text")]
        public string? CensoredText { get; set; }
    }

    // -----------------------------------------------------------------------
    // Request — trimis la API
    // -----------------------------------------------------------------------

    public class DetectRequest
    {
        // Textul de analizat
        [JsonPropertyName("text")]
        public string Text { get; set; } = "";

        // Limba forțată ("ro" / "en") sau null pentru auto-detecție
        [JsonPropertyName("lang")]
        public string? Lang { get; set; }

        // Dacă true, răspunsul include textul cenzurat cu '#'
        [JsonPropertyName("censor_hate")]
        public bool CensorHate { get; set; } = false;
    }
}
