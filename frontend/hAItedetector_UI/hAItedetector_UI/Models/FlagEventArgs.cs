using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace hAItedetector_UI.Models
{
    public class FlagEventArgs : EventArgs
    {
        public string Text { get; set; } = "";
        public string PredictedLabel { get; set; } = "";
        public string CorrectLabel { get; set; } = "";
    }
}
