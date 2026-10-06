"""Split a (gzipped) LHE file into chunks of N events (header copied)."""
import gzip
import sys

src, prefix, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
op = gzip.open if src.endswith(".gz") else open
with op(src, "rt") as f:
    header, line = [], f.readline()
    while "<event" not in line:
        header.append(line)
        line = f.readline()
    k, cnt, out = 0, 0, None
    while line:
        if "<event" in line:
            if cnt % n == 0:
                if out:
                    out.write("</LesHouchesEvents>\n")
                    out.close()
                out = gzip.open(f"{prefix}_{k:03d}.lhe.gz", "wt")
                out.writelines(header)
                k += 1
            cnt += 1
        if "</LesHouchesEvents>" not in line:
            out.write(line)
        line = f.readline()
    out.write("</LesHouchesEvents>\n")
    out.close()
print(f"{cnt} events -> {k} chunks")
