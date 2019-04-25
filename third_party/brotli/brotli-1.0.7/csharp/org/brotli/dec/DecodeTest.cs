/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>
	/// Tests for
	/// <see cref="Decode"/>
	/// .
	/// </summary>
	public class DecodeTest
	{
		/// <exception cref="System.IO.IOException"/>
		private byte[] Decompress(byte[] data, bool byByte)
		{
			byte[] buffer = new byte[65536];
			System.IO.MemoryStream input = new System.IO.MemoryStream(data);
			System.IO.MemoryStream output = new System.IO.MemoryStream();
			Org.Brotli.Dec.BrotliInputStream brotliInput = new Org.Brotli.Dec.BrotliInputStream(input);
			if (byByte)
			{
				byte[] oneByte = new byte[1];
				while (true)
				{
					int next = brotliInput.ReadByte();
					if (next == -1)
					{
						break;
					}
					oneByte[0] = unchecked((byte)next);
					output.Write(oneByte, 0, 1);
				}
			}
			else
			{
				while (true)
				{
					int len = brotliInput.Read(buffer, 0, buffer.Length);
					if (len <= 0)
					{
						break;
					}
					output.Write(buffer, 0, len);
				}
			}
			brotliInput.Close();
			return output.ToArray();
		}

		/// <exception cref="System.IO.IOException"/>
		private byte[] DecompressWithDictionary(byte[] data, byte[] dictionary)
		{
			byte[] buffer = new byte[65536];
			System.IO.MemoryStream input = new System.IO.MemoryStream(data);
			System.IO.MemoryStream output = new System.IO.MemoryStream();
			Org.Brotli.Dec.BrotliInputStream brotliInput = new Org.Brotli.Dec.BrotliInputStream(input, Org.Brotli.Dec.BrotliInputStream.DefaultInternalBufferSize, dictionary);
			while (true)
			{
				int len = brotliInput.Read(buffer, 0, buffer.Length);
				if (len <= 0)
				{
					break;
				}
				output.Write(buffer, 0, len);
			}
			brotliInput.Close();
			return output.ToArray();
		}

		/// <exception cref="System.IO.IOException"/>
		private void CheckDecodeResourceWithDictionary(string expected, string compressed, string dictionary)
		{
			byte[] expectedBytes = Org.Brotli.Dec.Transform.ReadUniBytes(expected);
			byte[] compressedBytes = Org.Brotli.Dec.Transform.ReadUniBytes(compressed);
			byte[] dictionaryBytes = Org.Brotli.Dec.Transform.ReadUniBytes(dictionary);
			byte[] actual = DecompressWithDictionary(compressedBytes, dictionaryBytes);
			NUnit.Framework.Assert.AreEqual(expectedBytes, actual);
		}

		/// <exception cref="System.IO.IOException"/>
		private void CheckDecodeResource(string expected, string compressed)
		{
			byte[] expectedBytes = Org.Brotli.Dec.Transform.ReadUniBytes(expected);
			byte[] compressedBytes = Org.Brotli.Dec.Transform.ReadUniBytes(compressed);
			byte[] actual = Decompress(compressedBytes, false);
			NUnit.Framework.Assert.AreEqual(expectedBytes, actual);
			byte[] actualByByte = Decompress(compressedBytes, true);
			NUnit.Framework.Assert.AreEqual(expectedBytes, actualByByte);
		}

		/// <exception cref="System.IO.IOException"/>
		[NUnit.Framework.Test]
		public virtual void TestEmpty()
		{
			CheckDecodeResource(string.Empty, "\u0006");
		}

		/// <exception cref="System.IO.IOException"/>
		[NUnit.Framework.Test]
		public virtual void TestX()
		{
			CheckDecodeResource("X", "\u000B\u0000\u0080X\u0003");
		}

		/// <exception cref="System.IO.IOException"/>
		[NUnit.Framework.Test]
		public virtual void TestX10Y10()
		{
			CheckDecodeResource("XXXXXXXXXXYYYYYYYYYY", "\u001B\u0013\u0000\u0000\u00A4\u00B0\u00B2\u00EA\u0081G\u0002\u008A");
		}

		/// <exception cref="System.IO.IOException"/>
		[NUnit.Framework.Test]
		public virtual void TestX64()
		{
			CheckDecodeResource("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", "\u001B\u003F\u0000\u0000$\u00B0\u00E2\u0099\u0080\u0012");
		}

		/// <exception cref="System.IO.IOException"/>
		[NUnit.Framework.Test]
		public virtual void TestUkkonooa()
		{
			CheckDecodeResource("ukko nooa, ukko nooa oli kunnon mies, kun han meni saunaan, " + "pisti laukun naulaan, ukko nooa, ukko nooa oli kunnon mies.", "\u001Bv\u0000\u0000\u0014J\u00AC\u009Bz\u00BD\u00E1\u0097\u009D\u007F\u008E\u00C2\u0082" + "6\u000E\u009C\u00E0\u0090\u0003\u00F7\u008B\u009E8\u00E6\u00B6\u0000\u00AB\u00C3\u00CA"
				 + "\u00A0\u00C2\u00DAf6\u00DC\u00CD\u0080\u008D.!\u00D7n\u00E3\u00EAL\u00B8\u00F0\u00D2" + "\u00B8\u00C7\u00C2pM:\u00F0i~\u00A1\u00B8Es\u00AB\u00C4W\u001E");
		}

		/// <exception cref="System.IO.IOException"/>
		[NUnit.Framework.Test]
		public virtual void TestMonkey()
		{
			CheckDecodeResource("znxcvnmz,xvnm.,zxcnv.,xcn.z,vn.zvn.zxcvn.,zxcn.vn.v,znm.,vnzx.,vnzxc.vn.z,vnz.,nv.z,nvmz" + "xc,nvzxcvcnm.,vczxvnzxcnvmxc.zmcnvzm.,nvmc,nzxmc,vn.mnnmzxc,vnxcnmv,znvzxcnmv,.xcnvm,zxc" + "nzxv.zx,qweryweurqioweupropqwutioweupqrioweutiopweuriopweuriopqwurioputiopqwuriowuqeriou"
				 + "pqweropuweropqwurweuqriopuropqwuriopuqwriopuqweopruioqweurqweuriouqweopruioupqiytioqtyio" + "wtyqptypryoqweutioioqtweqruowqeytiowquiourowetyoqwupiotweuqiorweuqroipituqwiorqwtioweuri" + "ouytuioerytuioweryuitoweytuiweyuityeruirtyuqriqweuropqweiruioqweurioqwuerioqwyuituierwot"
				 + "ueryuiotweyrtuiwertyioweryrueioqptyioruyiopqwtjkasdfhlafhlasdhfjklashjkfhasjklfhklasjdfh" + "klasdhfjkalsdhfklasdhjkflahsjdkfhklasfhjkasdfhasfjkasdhfklsdhalghhaf;hdklasfhjklashjklfa" + "sdhfasdjklfhsdjklafsd;hkldadfjjklasdhfjasddfjklfhakjklasdjfkl;asdjfasfljasdfhjklasdfhjka"
				 + "ghjkashf;djfklasdjfkljasdklfjklasdjfkljasdfkljaklfj", "\u001BJ\u0003\u0000\u008C\u0094n\u00DE\u00B4\u00D7\u0096\u00B1x\u0086\u00F2-\u00E1\u001A" + "\u00BC\u000B\u001C\u00BA\u00A9\u00C7\u00F7\u00CCn\u00B2B4QD\u008BN\u0013\b\u00A0\u00CDn"
				 + "\u00E8,\u00A5S\u00A1\u009C],\u001D#\u001A\u00D2V\u00BE\u00DB\u00EB&\u00BA\u0003e|\u0096j" + "\u00A2v\u00EC\u00EF\u0087G3\u00D6\'\u000Ec\u0095\u00E2\u001D\u008D,\u00C5\u00D1(\u009F`" + "\u0094o\u0002\u008B\u00DD\u00AAd\u0094,\u001E;e|\u0007EZ\u00B2\u00E2\u00FCI\u0081,\u009F"
				 + "@\u00AE\u00EFh\u0081\u00AC\u0016z\u000F\u00F5;m\u001C\u00B9\u001E-_\u00D5\u00C8\u00AF^" + "\u0085\u00AA\u0005\u00BESu\u00C2\u00B0\"\u008A\u0015\u00C6\u00A3\u00B1\u00E6B\u0014" + "\u00F4\u0084TS\u0019_\u00BE\u00C3\u00F2\u001D\u00D1\u00B7\u00E5\u00DD\u00B6\u00D9#\u00C6"
				 + "\u00F6\u009F\u009E\u00F6Me0\u00FB\u00C0qE\u0004\u00AD\u0003\u00B5\u00BE\u00C9\u00CB" + "\u00FD\u00E2PZFt\u0004\r\u00FF \u0004w\u00B2m\'\u00BFG\u00A9\u009D\u001B\u0096,b\u0090#" + "\u008B\u00E0\u00F8\u001D\u00CF\u00AF\u001D=\u00EE\u008A\u00C8u#f\u00DD\u00DE\u00D6m"
				 + "\u00E3*\u0082\u008Ax\u008A\u00DB\u00E6 L\u00B7\\c\u00BA0\u00E3?\u00B6\u00EE\u008C\"" + "\u00A2*\u00B0\"\n\u0099\u00FF=bQ\u00EE\b\u00F6=J\u00E4\u00CC\u00EF\"\u0087\u0011\u00E2" + "\u0083(\u00E4\u00F5\u008F5\u0019c[\u00E1Z\u0092s\u00DD\u00A1P\u009D8\\\u00EB\u00B5\u0003"
				 + "jd\u0090\u0094\u00C8\u008D\u00FB/\u008A\u0086\"\u00CC\u001D\u0087\u00E0H\n\u0096w\u00909" + "\u00C6##H\u00FB\u0011GV\u00CA \u00E3B\u0081\u00F7w2\u00C1\u00A5\\@!e\u0017@)\u0017\u0017" + "lV2\u00988\u0006\u00DC\u0099M3)\u00BB\u0002\u00DFL&\u0093l\u0017\u0082\u0086 \u00D7"
				 + "\u0003y}\u009A\u0000\u00D7\u0087\u0000\u00E7\u000Bf\u00E3Lfqg\b2\u00F9\b>\u00813\u00CD" + "\u0017r1\u00F0\u00B8\u0094RK\u00901\u008Eh\u00C1\u00EF\u0090\u00C9\u00E5\u00F2a\tr%" + "\u00AD\u00EC\u00C5b\u00C0\u000B\u0012\u0005\u00F7\u0091u\r\u00EEa..\u0019\t\u00C2\u0003"
				);
		}

		/// <exception cref="System.IO.IOException"/>
		[NUnit.Framework.Test]
		public virtual void TestFox()
		{
			CheckDecodeResource("The quick brown fox jumps over the lazy dog", "\u001B*\u0000\u0000\u0004\u0004\u00BAF:\u0085\u0003\u00E9\u00FA\f\u0091\u0002H\u0011," + "\u00F3\u008A:\u00A3V\u007F\u001A\u00AE\u00BF\u00A4\u00AB\u008EM\u00BF\u00ED\u00E2\u0004K"
				 + "\u0091\u00FF\u0087\u00E9\u001E");
		}

		/// <exception cref="System.IO.IOException"/>
		[NUnit.Framework.Test]
		public virtual void TestFoxFox()
		{
			CheckDecodeResourceWithDictionary("The quick brown fox jumps over the lazy dog", "\u001B*\u0000\u0000 \u0000\u00C2\u0098\u00B0\u00CA\u0001", "The quick brown fox jumps over the lazy dog");
		}

		[NUnit.Framework.Test]
		public virtual void TestUtils()
		{
			new Org.Brotli.Dec.Context();
			new Org.Brotli.Dec.Decode();
			new Org.Brotli.Dec.Dictionary();
			new Org.Brotli.Dec.Huffman();
			new Org.Brotli.Dec.Prefix();
		}
	}
}
