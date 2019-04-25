/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>
	/// Tests for
	/// <see cref="Dictionary"/>
	/// .
	/// </summary>
	public class DictionaryTest
	{
		private static long Crc64(byte[] data)
		{
			long crc = -1;
			for (int i = 0; i < data.Length; ++i)
			{
				long c = (crc ^ (long)(data[i] & unchecked((int)(0xFF)))) & unchecked((int)(0xFF));
				for (int k = 0; k < 8; k++)
				{
					c = ((long)(((ulong)c) >> 1)) ^ (-(c & 1L) & -3932672073523589310L);
				}
				crc = c ^ ((long)(((ulong)crc) >> 8));
			}
			return ~crc;
		}

		[NUnit.Framework.Test]
		public virtual void TestGetData()
		{
			NUnit.Framework.Assert.AreEqual(37084801881332636L, Crc64(Org.Brotli.Dec.Dictionary.GetData()));
		}
	}
}
