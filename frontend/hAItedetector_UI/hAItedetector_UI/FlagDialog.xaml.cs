using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace hAItedetector.UI
{
    public partial class FlagDialog : Window
    {
        public string SelectedLabel { get; private set; } = "normal";

        public FlagDialog(string currentLabel)
        {
            InitializeComponent();

            // preselectăm eticheta curentă
            foreach (ComboBoxItem item in CmbLabel.Items)
            {
                if (item.Tag as string == currentLabel)
                {
                    item.IsSelected = true;
                    break;
                }
            }
        }

        private void BtnConfirm_Click(object sender, RoutedEventArgs e)
        {
            if (CmbLabel.SelectedItem is ComboBoxItem selected)
                SelectedLabel = selected.Tag as string ?? "normal";

            DialogResult = true;
            Close();
        }

        private void BtnCancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}