// SourceAFIS comparator driver for the fingerprint-from-image FVC benchmark.
//
//   extract <work_dir> <dpi>
//       Reads <work_dir>/images.txt (lines: "index width height file.raw"; raw =
//       row-major 8-bit grayscale, 0 = black ridge) and writes <index>.tpl
//       (SourceAFIS CBOR template) or <index>.fail.
//   match <work_dir> <pairs.txt> <scores.txt>
//       Reads "i j" lines, scores FingerprintMatcher(template_i).Match(template_j),
//       writes one score per line ("-inf" when a template is missing).
//
// Both modes print one JSON line with in-process timing (excludes process
// start-up and JIT warm-up, which the first item absorbs and is excluded).
using System.Diagnostics;
using System.Globalization;
using SourceAFIS;

static int Fail(string message)
{
    Console.Error.WriteLine(message);
    return 2;
}

if (args.Length < 1)
{
    return Fail("usage: extract <work_dir> <dpi> | match <work_dir> <pairs.txt> <scores.txt>");
}
var inv = CultureInfo.InvariantCulture;

if (args[0] == "extract" && args.Length == 3)
{
    var dir = args[1];
    var dpi = double.Parse(args[2], inv);
    var lines = File.ReadAllLines(Path.Combine(dir, "images.txt"));
    var watch = new Stopwatch();
    int failed = 0, timed = 0;
    for (var k = 0; k < lines.Length; k++)
    {
        var parts = lines[k].Split(' ');
        var index = int.Parse(parts[0], inv);
        var width = int.Parse(parts[1], inv);
        var height = int.Parse(parts[2], inv);
        var pixels = File.ReadAllBytes(Path.Combine(dir, parts[3]));
        var start = watch.Elapsed;
        if (k > 0) watch.Start();
        try
        {
            var image = new FingerprintImage(width, height, pixels, new FingerprintImageOptions { Dpi = dpi });
            var template = new FingerprintTemplate(image);
            watch.Stop();
            File.WriteAllBytes(Path.Combine(dir, $"{index}.tpl"), template.ToByteArray());
        }
        catch (Exception exc)
        {
            watch.Stop();
            failed++;
            File.WriteAllText(Path.Combine(dir, $"{index}.fail"), exc.Message);
        }
        if (k > 0) timed++;
    }
    var ms = timed > 0 ? watch.Elapsed.TotalMilliseconds / timed : 0.0;
    Console.WriteLine($"{{\"mode\":\"extract\",\"images\":{lines.Length},\"failed\":{failed},\"ms_per_image\":{ms.ToString("R", inv)}}}");
    return 0;
}

if (args[0] == "match" && args.Length == 4)
{
    var dir = args[1];
    var cache = new Dictionary<int, FingerprintTemplate?>();
    FingerprintTemplate? Load(int index)
    {
        if (cache.TryGetValue(index, out var hit)) return hit;
        var path = Path.Combine(dir, $"{index}.tpl");
        var template = File.Exists(path) ? new FingerprintTemplate(File.ReadAllBytes(path)) : null;
        cache[index] = template;
        return template;
    }
    var pairs = File.ReadAllLines(args[2]);
    var output = new List<string>(pairs.Length);
    var watch = new Stopwatch();
    int timed = 0;
    for (var k = 0; k < pairs.Length; k++)
    {
        var parts = pairs[k].Split(' ');
        var probe = Load(int.Parse(parts[0], inv));
        var candidate = Load(int.Parse(parts[1], inv));
        if (probe is null || candidate is null)
        {
            output.Add("-inf");
            continue;
        }
        if (k > 0) watch.Start();
        // Matcher construction is part of a one-to-one comparison cost.
        var score = new FingerprintMatcher(probe).Match(candidate);
        watch.Stop();
        if (k > 0) timed++;
        output.Add(score.ToString("R", inv));
    }
    File.WriteAllLines(args[3], output);
    var us = timed > 0 ? watch.Elapsed.TotalMilliseconds * 1000.0 / timed : 0.0;
    Console.WriteLine($"{{\"mode\":\"match\",\"pairs\":{pairs.Length},\"us_per_pair\":{us.ToString("R", inv)}}}");
    return 0;
}

return Fail("usage: extract <work_dir> <dpi> | match <work_dir> <pairs.txt> <scores.txt>");
