# This script saves each interval in the selected IntervalTier of a TextGrid to a separate WAV sound file.
# The source sound must be a LongSound object, and both the TextGrid and 
# the LongSound must have identical names and they have to be selected 
# before running the script.
# Files are named with the corresponding interval labels (plus a running index number when necessary).
#
# NOTE: You have to take care yourself that the interval labels do not contain forbidden characters!!!!
# 
# This script is distributed under the GNU General Public License.
# Copyright 8.3.2002 Mietta Lennes
#

form Save intervals to small WAV sound files
	comment Which IntervalTier in this TextGrid would you like to process?
	integer Tier 1
	#comment Starting and ending at which interval? 
	#integer Start_from 1
	#integer End_at_(0=last) 0
	boolean Exclude_empty_labels 1
	boolean Exclude_intervals_labeled_as_xxx 1
	boolean Exclude_intervals_starting_with_dot_(.) 1
	comment Give a small margin for the files if you like:
	positive Margin_(seconds) 0.05
	comment Give the folder where to save the sound files:
	sentence Folder /home/matias/Desktop/PRUEBA_GER/
	comment Give an optional prefix for all filenames:
	sentence Prefix(por_ej.Amperes)
	comment Give an optional suffix for all filenames (.wav will be added anyway):
	sentence Suffix1(por_ej.Fecha)
        sentence Suffix2 
endform

select Strings list
file_count = Get number of strings

for current_file from 1 to file_count
   select Strings list
   filename$ = Get string... current_file
   newLength = length (filename$) - length (".wav")
   newFilename$ = left$ (filename$, newLength)	
   soundname$ = replace$ (newFilename$, ".", "_", 0)
   gridname$ = soundname$
   dataave$ = mid$ (soundname$,3,28)
   datafile$ = right$ (soundname$,4)
   
start_from = 1
#--------------------------------------------------------  
#con esto le saco la primera letra que corresponde a presion
#para que use el mismo textgrid que el sound para extraer la presion
presion = 0
if left$(gridname$, 1)="p" 
  presion =1
endif
gridname$ = "s" + right$ (gridname$, newLength-1)	
presname$ =replace$ (soundname$, "s", "p", 0) 
#--------------------------------------------------------   

  
select TextGrid 'gridname$'
numberOfIntervals = Get number of intervals... tier
if start_from > numberOfIntervals
	exit There are not that many intervals in the IntervalTier!
endif
#if end_at > numberOfIntervals
	end_at = numberOfIntervals
#endif
#if end_at = numberOfIntervals
	end_at = numberOfIntervals
#end if


# Default values for variables
files = 0
intervalstart = 0
intervalend = 0
interval = 1
intname$ = ""
intervalfile$ = ""
endoffile = Get finishing time

# ask if the user wants to go through with saving all the files:
for interval from start_from to end_at
	xxx$ = Get label of interval... tier interval
	check = 0
	if xxx$ = "xxx" and exclude_intervals_labeled_as_xxx = 1
		check = 1
	endif
	if xxx$ = "" and exclude_empty_labels = 1
		check = 1
	endif
	if left$ (xxx$,1) = "." and exclude_intervals_starting_with_dot = 1
		check = 1
	endif
	if check = 0
	   files = files + 1
	endif
endfor
interval = 1
pause 'files' sound files will be saved. Continue?

# Loop through all intervals in the selected tier of the TextGrid
for interval from start_from to end_at
	

        select TextGrid 'gridname$'
	intname$ = ""
	intname$ = Get label of interval... tier interval
	check = 0
	if intname$ = "xxx" and exclude_intervals_labeled_as_xxx = 1
		check = 1
	endif
	if intname$ = "" and exclude_empty_labels = 1
		check = 1
	endif
	if left$ (intname$,1) = "." and exclude_intervals_starting_with_dot = 1
		check = 1
	endif
	if check = 0
		intervalstart = Get starting point... tier interval
			if intervalstart > margin
				intervalstart = intervalstart - margin
			else
				intervalstart = 0
			endif
	
		intervalend = Get end point... tier interval
			if intervalend < endoffile - margin
				intervalend = intervalend + margin
			else
				intervalend = endoffile
			endif


		select LongSound 'soundname$'
		Extract part... intervalstart intervalend no
		filename$ = intname$
		#para presion---------------------
		if presion=1
			suffix2$="'suffix2$'p"
			else
			suffix2$="'suffix2$'s"
		endif
		#------------------------------------
		intervalfile$ = "'folder$'" + "'prefix$'" + "'filename$'_"  + "'dataave$'_"+ "['datafile$']_" +  "'suffix1$'"+"'suffix2$'" + ".wav"
		indexnumber = 0
		while fileReadable (intervalfile$)
			indexnumber = indexnumber + 1
			intervalfile$ = "'folder$'" + "'prefix$'" + "'filename$'_"  + "'dataave$'_"+ "['datafile$']_" +  "'suffix1$'"+"'suffix2$'_'indexnumber'" + ".wav"
		endwhile
		#---------------------
		suffix2$=left$(suffix2$,length(suffix2$)-1)
		#------------------------------------
		Write to WAV file... 'intervalfile$'
		Remove

		select LongSound 'presname$'
		Extract part... intervalstart intervalend no
		filename$ = intname$
		#para presion---------------------
		
			suffix2$="'suffix2$'p"
			
			
		#------------------------------------
		intervalfile$ = "'folder$'" + "'prefix$'" + "'filename$'_"  + "'dataave$'_"+ "['datafile$']_" +  "'suffix1$'"+"'suffix2$'" + ".wav"
		indexnumber = 0
		while fileReadable (intervalfile$)
			indexnumber = indexnumber + 1
			intervalfile$ = "'folder$'" + "'prefix$'" + "'filename$'_"  + "'dataave$'_"+ "['datafile$']_" +  "'suffix1$'"+"'suffix2$'_'indexnumber'" + ".wav"
		endwhile
		#---------------------
		suffix2$=left$(suffix2$,length(suffix2$)-1)
		#------------------------------------
		Write to WAV file... 'intervalfile$'
		Remove





	endif
endfor

endfor