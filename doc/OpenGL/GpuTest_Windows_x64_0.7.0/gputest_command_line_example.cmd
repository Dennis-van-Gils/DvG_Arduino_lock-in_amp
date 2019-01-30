rem GpuTest command line parameters:
rem /test=<test_name> where test_name = fur | tess | gi | pixmark_piano | pixmark_volplosion | triangle | plot3d
rem /width=<window_width>
rem /height=<window_height>
rem /fullscreen
rem /msaa=<samples> where samples can be 0, 2, 4, 8
rem /benchmark : sets the benchmarking mode
rem /benchmark_duration_ms=<duration_in_ms> : benchmark duration in milliseconds.
rem /no_scorebox : does not display the score box at the end of the benchmark.
rem /no_log_score : does not write score result in the log file at the end of the benchmark.
rem /glinfo : write detailed OpenGL information in the log file
rem /debug_log_frame_data : write per frame data (frame number, elapsed time, opengl error code) in the log file
rem /display_info : displays test information and GPU monitoring (Windows only).

rem Simple test:
start GpuTest.exe /test=fur /width=1024 /height=640

rem Benchmark:
rem start GpuTest.exe /test=fur /width=1280 /height=720 /msaa=4 /benchmark /benchmark_duration_ms=60000
