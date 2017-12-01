## Praat script by Kevin Ryan 9/05
## Below: the user is asked for a directory (the default below is the path for my own desktop;
## you will probably want to change that), a file extension, and an optional substring to match
## in the filenames (leaving this blank will get all the files of the given type)

form Read all files from directory and annotate silences and sounds.
   sentence Source_directory 
   sentence File_name_or_initial_substring s
   sentence File_extension .wav
   
   real minimun_pitch(Hz) 100
   real timpe_step(s) 0.0 (=auto)
   real silence_threshold(dB) -35
   real minimun_silence_duration 0.2
   real minimun_sounding_duration 0.1
   sentence Nombre_canto 
endform

## Below: collect all the files that match the search criteria and save them

Create Strings as file list... list 'source_directory$'/'file_name_or_initial_substring$'*'file_extension$'
head_words = selected("Strings")
file_count = Get number of strings

## Below: loop through the list of files, extracting each name and reading it into the Objects list

for current_file from 1 to file_count
   select Strings list
   filename$ = Get string... current_file
   Read from file... 'source_directory$'/'filename$'
  
   To TextGrid (silences)... minimun_pitch timpe_step silence_threshold minimun_silence_duration minimun_sounding_duration "" 'nombre_canto$'
   Open long sound file... 'source_directory$'/'filename$'

endfor

## remove the temporary sonf files not needed
for current_file from 1 to file_count
   select 'head_words'
   filename$ = Get string... current_file
   newLength = length (filename$) - length (".wav")
   newFilename$ = left$ (filename$, newLength)	
   filename$ = replace$ (newFilename$, ".", "_", 0)
   select Sound 'filename$'
   Remove

endfor

## Finally, remove the temporary file list object (head_words) and report the number of files read
clearinfo
echo Done! 'file_count' files read.'newline$'.





