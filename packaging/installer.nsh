!macro customInit
  SetRegView 64
  ReadRegStr $R0 HKLM "SOFTWARE\Microsoft\Windows NT\CurrentVersion" "CurrentBuildNumber"
  IntCmp $R0 22000 steno_supported steno_unsupported steno_supported
  steno_unsupported:
    MessageBox MB_ICONSTOP "Steno requires Windows 11 (64-bit) or later."
    Abort
  steno_supported:
!macroend
