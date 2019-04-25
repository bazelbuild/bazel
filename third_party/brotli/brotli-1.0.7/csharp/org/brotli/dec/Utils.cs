/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>A set of utility methods.</summary>
	internal sealed class Utils
	{
		private static readonly byte[] ByteZeroes = new byte[1024];

		private static readonly int[] IntZeroes = new int[1024];

		/// <summary>Fills byte array with zeroes.</summary>
		/// <remarks>
		/// Fills byte array with zeroes.
		/// <p> Current implementation uses
		/// <see cref="System.Array.Copy(object, int, object, int, int)"/>
		/// , so it should be used for length not
		/// less than 16.
		/// </remarks>
		/// <param name="dest">array to fill with zeroes</param>
		/// <param name="offset">the first byte to fill</param>
		/// <param name="length">number of bytes to change</param>
		internal static void FillWithZeroes(byte[] dest, int offset, int length)
		{
			int cursor = 0;
			while (cursor < length)
			{
				int step = System.Math.Min(cursor + 1024, length) - cursor;
				System.Array.Copy(ByteZeroes, 0, dest, offset + cursor, step);
				cursor += step;
			}
		}

		/// <summary>Fills int array with zeroes.</summary>
		/// <remarks>
		/// Fills int array with zeroes.
		/// <p> Current implementation uses
		/// <see cref="System.Array.Copy(object, int, object, int, int)"/>
		/// , so it should be used for length not
		/// less than 16.
		/// </remarks>
		/// <param name="dest">array to fill with zeroes</param>
		/// <param name="offset">the first item to fill</param>
		/// <param name="length">number of item to change</param>
		internal static void FillWithZeroes(int[] dest, int offset, int length)
		{
			int cursor = 0;
			while (cursor < length)
			{
				int step = System.Math.Min(cursor + 1024, length) - cursor;
				System.Array.Copy(IntZeroes, 0, dest, offset + cursor, step);
				cursor += step;
			}
		}
	}
}
