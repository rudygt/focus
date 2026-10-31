# Activate the top window on the left or right monitor.
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File focus.ps1 -Side left|right|list
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File focus.ps1 -Serve
# Serve mode prints "ready", then answers each stdin line (left, right, list) with one stdout line.
param([ValidateSet('left', 'right', 'list')] [string]$Side = 'list', [switch]$Serve)
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$src = @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public static class FocusSwitch {
    delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    delegate bool MonitorEnumProc(IntPtr hMon, IntPtr hdc, ref RECT r, IntPtr d);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }

    [DllImport("user32.dll")] static extern bool EnumWindows(EnumWindowsProc cb, IntPtr lp);
    [DllImport("user32.dll")] static extern bool EnumDisplayMonitors(IntPtr hdc, IntPtr clip, MonitorEnumProc cb, IntPtr d);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] static extern bool IsIconic(IntPtr h);
    [DllImport("user32.dll")] static extern int GetWindowTextLength(IntPtr h);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] static extern int GetWindowLong(IntPtr h, int i);
    [DllImport("user32.dll")] static extern IntPtr MonitorFromWindow(IntPtr h, uint flags);
    [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
    [DllImport("dwmapi.dll")] static extern int DwmGetWindowAttribute(IntPtr h, int attr, out int v, int size);

    const int GWL_EXSTYLE = -20, WS_EX_TOOLWINDOW = 0x80, DWMWA_CLOAKED = 14;
    const uint MONITOR_DEFAULTTONEAREST = 2;
    const byte VK_MENU = 0x12;

    public class Mon { public IntPtr H; public RECT R; }
    public class Win { public IntPtr H; public string Title; public int Mon; }

    // Sorted left to right by screen x.
    public static List<Mon> Monitors() {
        var list = new List<Mon>();
        EnumDisplayMonitors(IntPtr.Zero, IntPtr.Zero,
            delegate(IntPtr h, IntPtr dc, ref RECT r, IntPtr d) { list.Add(new Mon { H = h, R = r }); return true; },
            IntPtr.Zero);
        list.Sort(delegate(Mon a, Mon b) { return a.R.Left.CompareTo(b.R.Left); });
        return list;
    }

    // Top-level windows in z-order, topmost first, so the first hit per monitor is its most recently active window.
    public static List<Win> Windows(List<Mon> mons) {
        var list = new List<Win>();
        EnumWindows(delegate(IntPtr h, IntPtr lp) {
            if (!IsWindowVisible(h) || IsIconic(h)) return true;
            if ((GetWindowLong(h, GWL_EXSTYLE) & WS_EX_TOOLWINDOW) != 0) return true;
            int cloaked;
            if (DwmGetWindowAttribute(h, DWMWA_CLOAKED, out cloaked, 4) == 0 && cloaked != 0) return true;
            int n = GetWindowTextLength(h);
            if (n == 0) return true;
            var sb = new StringBuilder(n + 1);
            GetWindowText(h, sb, n + 1);
            string title = sb.ToString();
            if (title == "Program Manager") return true;
            IntPtr hm = MonitorFromWindow(h, MONITOR_DEFAULTTONEAREST);
            int idx = mons.FindIndex(delegate(Mon m) { return m.H == hm; });
            list.Add(new Win { H = h, Title = title, Mon = idx });
            return true;
        }, IntPtr.Zero);
        return list;
    }

    public static bool Activate(IntPtr h) {
        if (GetForegroundWindow() == h) return true;
        // A synthetic ALT tap makes Windows treat this process as having recent input, which SetForegroundWindow requires.
        keybd_event(VK_MENU, 0, 0, UIntPtr.Zero);
        keybd_event(VK_MENU, 0, 2, UIntPtr.Zero);
        SetForegroundWindow(h);
        System.Threading.Thread.Sleep(80);
        return GetForegroundWindow() == h;
    }
}
'@

# Compile once per source version; a fresh csc run costs ~1s per call otherwise.
$hash = [BitConverter]::ToString([System.Security.Cryptography.MD5]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($src))).Replace('-', '').Substring(0, 12)
$dll = Join-Path $env:TEMP "focus-switch-$hash.dll"
if (-not (Test-Path $dll)) { Add-Type -TypeDefinition $src -OutputAssembly $dll }
Add-Type -Path $dll

function Describe-Layout {
    $mons = [FocusSwitch]::Monitors()
    $wins = [FocusSwitch]::Windows($mons)
    $lines = @()
    for ($i = 0; $i -lt $mons.Count; $i++) {
        $r = $mons[$i].R
        $lines += "monitor {0}: x={1}..{2} y={3}..{4}" -f $i, $r.Left, $r.Right, $r.Top, $r.Bottom
    }
    foreach ($w in $wins) { $lines += "  [mon {0}] {1}" -f $w.Mon, $w.Title }
    $lines -join "`n"
}

function Switch-Focus([string]$To) {
    $mons = [FocusSwitch]::Monitors()
    $wins = [FocusSwitch]::Windows($mons)
    if ($mons.Count -lt 2) { return "need two monitors, found $($mons.Count)" }
    $target = if ($To -eq 'left') { 0 } else { $mons.Count - 1 }
    $win = $wins | Where-Object { $_.Mon -eq $target } | Select-Object -First 1
    if (-not $win) { return "no window on $To monitor" }
    $ok = [FocusSwitch]::Activate($win.H)
    "{0} {1}: {2}" -f $(if ($ok) { 'focused' } else { 'FAILED' }), $To, $win.Title
}

if ($Serve) {
    $out = [Console]::Out
    $out.WriteLine('ready'); $out.Flush()
    while ($null -ne ($line = [Console]::In.ReadLine())) {
        $reply = switch ($line.Trim()) {
            'left' { Switch-Focus 'left' }
            'right' { Switch-Focus 'right' }
            'list' { Describe-Layout }
            default { "unknown command: $line" }
        }
        $out.WriteLine(($reply -replace "`n", ' | ')); $out.Flush()
    }
    exit 0
}

if ($Side -eq 'list') { Write-Output (Describe-Layout); exit 0 }
$result = Switch-Focus $Side
Write-Output $result
exit $(if ($result.StartsWith('focused')) { 0 } else { 1 })
