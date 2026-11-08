// MicroMind Setup — installs microai.exe + corpus.txt and adds it to the user PATH.
// No dependencies required. Run: MicroMindSetup.exe
using System;
using System.IO;
using System.Reflection;
using Microsoft.Win32;

class MicroMindSetup
{
    static int Main()
    {
        try
        {
            string destDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs", "MicroMind");
            Directory.CreateDirectory(destDir);

            Assembly asm = Assembly.GetExecutingAssembly();

            // payload 1: microai.exe (embedded resource, or file beside setup)
            InstallPayload(asm, "microai.exe", Path.Combine(destDir, "microai.exe"));
            // payload 2: corpus.txt training data
            InstallPayload(asm, "corpus.txt", Path.Combine(destDir, "corpus.txt"));

            // add destDir to the USER PATH (registry — no admin needed)
            using (RegistryKey key = Registry.CurrentUser.OpenSubKey("Environment", true))
            {
                string path = key.GetValue("Path", "") as string ?? "";
                var parts = new System.Collections.Generic.List<string>(path.Split(';'));
                bool found = false;
                foreach (var p in parts)
                    if (string.Equals(p.TrimEnd('\\'), destDir.TrimEnd('\\'), StringComparison.OrdinalIgnoreCase))
                        found = true;
                if (!found)
                {
                    parts.Add(destDir);
                    key.SetValue("Path", string.Join(";", parts), RegistryValueKind.ExpandString);
                    Console.WriteLine("added to user PATH: " + destDir);
                }
                else Console.WriteLine("already on PATH: " + destDir);
            }

            // notify the shell so new terminals pick up PATH
            try
            {
                IntPtr result;
                SendMessageTimeout(new IntPtr(0xFFFF), 0x1A, IntPtr.Zero, "Environment",
                    2, 5000, out result);
            }
            catch { }

            string corpus = Path.Combine(destDir, "corpus.txt");
            Console.WriteLine();
            Console.WriteLine("Done. Open a NEW terminal and run:");
            Console.WriteLine("    microai train \"" + corpus + "\"");
            Console.WriteLine("    microai gen 500 \"the law\"");
            Console.WriteLine("    microai repl");
            Console.WriteLine();
            Console.WriteLine("(current terminals won't see 'microai' until restarted)");
            Console.Write("Press Enter to exit...");
            Console.ReadLine();
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine("install failed: " + ex.Message);
            Console.WriteLine(ex);
            Console.Write("Press Enter to exit...");
            Console.ReadLine();
            return 1;
        }
    }

    static void InstallPayload(Assembly asm, string name, string dest)
    {
        Stream res = asm.GetManifestResourceStream(name);
        Stream beside = null;
        if (res == null)
        {
            string p = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, name);
            if (File.Exists(p)) beside = File.OpenRead(p);
        }
        try
        {
            Stream input = res ?? (Stream)beside;
            if (input == null) { Console.WriteLine("error: " + name + " payload not found"); Environment.Exit(1); }
            using (FileStream fs = File.Create(dest)) input.CopyTo(fs);
            Console.WriteLine("installed: " + dest);
        }
        finally { if (beside != null) beside.Dispose(); if (res != null) res.Dispose(); }
    }

    [System.Runtime.InteropServices.DllImport("user32.dll", SetLastError = true,
        CharSet = System.Runtime.InteropServices.CharSet.Auto)]
    static extern IntPtr SendMessageTimeout(IntPtr hWnd, uint msg, IntPtr wParam,
        string lParam, uint flags, uint timeout, out IntPtr result);
}
