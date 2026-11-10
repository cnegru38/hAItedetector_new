using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace hAItedetector_UI.Models
{
    public class MessageItem
    {
        public string Text { get; set; } = "";
        public string Label { get; set; } = "—";
        public string Score { get; set; } = "—";
        public Dictionary<string, double> Scores { get; set; } = new();
        public string SenderName { get; set; } = "";
    }
}
