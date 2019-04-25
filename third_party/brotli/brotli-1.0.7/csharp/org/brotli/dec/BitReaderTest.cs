/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>
	/// Tests for
	/// <see cref="BitReader"/>
	/// .
	/// </summary>
	public class BitReaderTest
	{
		[NUnit.Framework.Test]
		public virtual void TestReadAfterEos()
		{
			Org.Brotli.Dec.BitReader reader = new Org.Brotli.Dec.BitReader();
			Org.Brotli.Dec.BitReader.Init(reader, new System.IO.MemoryStream(new byte[1]));
			Org.Brotli.Dec.BitReader.ReadBits(reader, 9);
			try
			{
				Org.Brotli.Dec.BitReader.CheckHealth(reader, false);
			}
			catch (Org.Brotli.Dec.BrotliRuntimeException)
			{
				// This exception is expected.
				return;
			}
			NUnit.Framework.Assert.Fail("BrotliRuntimeException should have been thrown by BitReader.checkHealth");
		}
	}
}
