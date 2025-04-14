' VBScript launcher for 3lacks Scanner
' This script launches the 3lacks Scanner in a new console window

' Get the path to the current directory
Set fso = CreateObject("Scripting.FileSystemObject")
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Get the path to the PyBloat directory
pybloatDir = currentDir & "\Py Bloat"

' Get the path to the terminal.py script
terminalPath = pybloatDir & "\terminal.py"

' Check if the PyBloat directory exists
If Not fso.FolderExists(pybloatDir) Then
    MsgBox "Error: PyBloat directory not found at " & pybloatDir, vbExclamation, "3lacks Scanner"
    WScript.Quit
End If

' Check if the terminal.py script exists
If Not fso.FileExists(terminalPath) Then
    MsgBox "Error: terminal.py not found in " & pybloatDir, vbExclamation, "3lacks Scanner"
    WScript.Quit
End If

' Get the path to the Python executable
pythonExe = "python"

' Create a new console window and run the terminal.py script
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = pybloatDir
shell.Run "cmd /k " & pythonExe & " " & terminalPath, 1, False
