/*
 * plist - An open source library to parse and generate property lists
 * Copyright (C) 2014 Daniel Dreibrodt
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package com.dd.plist;

import org.w3c.dom.*;
import org.xml.sax.EntityResolver;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;

/**
 * Parses XML property lists.
 *
 * @author Daniel Dreibrodt
 */
public class XMLPropertyListParser {

    /**
     * Instantiation is prohibited by outside classes.
     */
    protected XMLPropertyListParser() {
        /** empty **/
    }

    private static DocumentBuilderFactory docBuilderFactory = null;

    /**
     * Initialize the document builder factory so that it can be reused and does not need to
     * be reinitialized for each parse action.
     */
    private static synchronized void initDocBuilderFactory() {
        docBuilderFactory = DocumentBuilderFactory.newInstance();
        docBuilderFactory.setIgnoringComments(true);
        docBuilderFactory.setCoalescing(true);
    }

    /**
     * Gets a DocumentBuilder to parse a XML property list.
     * As DocumentBuilders are not thread-safe a new DocBuilder is generated for each request.
     *
     * @return A new DocBuilder that can parse property lists w/o an internet connection.
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     */
    private static synchronized DocumentBuilder getDocBuilder() throws ParserConfigurationException {
        if (docBuilderFactory == null)
            initDocBuilderFactory();
        DocumentBuilder docBuilder = docBuilderFactory.newDocumentBuilder();
        docBuilder.setEntityResolver(new EntityResolver() {
            public InputSource resolveEntity(String publicId, String systemId) {
                if ("-//Apple Computer//DTD PLIST 1.0//EN".equals(publicId) || // older publicId
                        "-//Apple//DTD PLIST 1.0//EN".equals(publicId)) { // newer publicId
                    // return a dummy, zero length DTD so we don't have to fetch
                    // it from the network.
                    return new InputSource(new ByteArrayInputStream(new byte[0]));
                }
                return null;
            }
        });
        return docBuilder;
    }

    /**
     * Parses a XML property list file.
     *
     * @param f The XML property list file.
     * @return The root object of the property list. This is usually a NSDictionary but can also be a NSArray.
     * @see javax.xml.parsers.DocumentBuilder#parse(java.io.File)
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     * @throws java.io.IOException If any IO error occurs while reading the file.
     * @throws org.xml.sax.SAXException If any parse error occurs.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static NSObject parse(File f) throws ParserConfigurationException, IOException, SAXException, PropertyListFormatException, ParseException {
        DocumentBuilder docBuilder = getDocBuilder();

        Document doc = docBuilder.parse(f);

        return parse(doc);
    }

    /**
     * Parses a XML property list from a byte array.
     *
     * @param bytes The byte array containing the property list's data.
     * @return The root object of the property list. This is usually a NSDictionary but can also be a NSArray.
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     * @throws java.io.IOException If any IO error occurs while reading the file.
     * @throws org.xml.sax.SAXException If any parse error occurs.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static NSObject parse(final byte[] bytes) throws ParserConfigurationException, ParseException, SAXException, PropertyListFormatException, IOException {
        ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
        return parse(bis);
    }

    /**
     * Parses a XML property list from an input stream.
     *
     * @param is The input stream pointing to the property list's data.
     * @return The root object of the property list. This is usually a NSDictionary but can also be a NSArray.
     * @see javax.xml.parsers.DocumentBuilder#parse(java.io.InputStream)
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     * @throws java.io.IOException If any IO error occurs while reading the file.
     * @throws org.xml.sax.SAXException If any parse error occurs.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static NSObject parse(InputStream is) throws ParserConfigurationException, IOException, SAXException, PropertyListFormatException, ParseException {
        DocumentBuilder docBuilder = getDocBuilder();

        Document doc = docBuilder.parse(is);

        return parse(doc);
    }

    /**
     * Parses a property list from an XML document.
     *
     * @param doc The XML document.
     * @return The root NSObject of the property list contained in the XML document.
     * @throws java.io.IOException If any IO error occurs while reading the file.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static NSObject parse(Document doc) throws PropertyListFormatException, IOException, ParseException {
        DocumentType docType = doc.getDoctype();
        if (docType == null) {
            if (!doc.getDocumentElement().getNodeName().equals("plist")) {
                throw new UnsupportedOperationException("The given XML document is not a property list.");
            }
        } else if (!docType.getName().equals("plist")) {
            throw new UnsupportedOperationException("The given XML document is not a property list.");
        }

        Node rootNode;

        if (doc.getDocumentElement().getNodeName().equals("plist")) {
            //Root element wrapped in plist tag
            List<Node> rootNodes = filterElementNodes(doc.getDocumentElement().getChildNodes());
            if (rootNodes.isEmpty()) {
                throw new PropertyListFormatException("The given XML property list has no root element!");
            } else if (rootNodes.size() == 1) {
                rootNode = rootNodes.get(0);
            } else {
                throw new PropertyListFormatException("The given XML property list has more than one root element!");
            }
        } else {
            //Root NSObject not wrapped in plist-tag
            rootNode = doc.getDocumentElement();
        }

        return parseObject(rootNode);
    }

    /**
     * Parses a node in the XML structure and returns the corresponding NSObject
     *
     * @param n The XML node.
     * @return The corresponding NSObject.
     * @throws java.io.IOException If any IO error occurs while parsing a Base64 encoded NSData object.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    private static NSObject parseObject(Node n) throws ParseException, IOException {
        String type = n.getNodeName();
        if (type.equals("dict")) {
            NSDictionary dict = new NSDictionary();
            List<Node> children = filterElementNodes(n.getChildNodes());
            for (int i = 0; i < children.size(); i += 2) {
                Node key = children.get(i);
                Node val = children.get(i + 1);

                String keyString = getNodeTextContents(key);

                dict.put(keyString, parseObject(val));
            }
            return dict;
        } else if (type.equals("array")) {
            List<Node> children = filterElementNodes(n.getChildNodes());
            NSArray array = new NSArray(children.size());
            for (int i = 0; i < children.size(); i++) {
                array.setValue(i, parseObject(children.get(i)));
            }
            return array;
        } else if (type.equals("true")) {
            return new NSNumber(true);
        } else if (type.equals("false")) {
            return new NSNumber(false);
        } else if (type.equals("integer")) {
            return new NSNumber(getNodeTextContents(n));
        } else if (type.equals("real")) {
            return new NSNumber(getNodeTextContents(n));
        } else if (type.equals("string")) {
            return new NSString(getNodeTextContents(n));
        } else if (type.equals("data")) {
            return new NSData(getNodeTextContents(n));
        } else if (type.equals("date")) {
            return new NSDate(getNodeTextContents(n));
        }
        return null;
    }

    /**
     * Returns all element nodes that are contained in a list of nodes.
     *
     * @param list The list of nodes to search.
     * @return The sub-list containing only nodes representing actual elements.
     */
    private static List<Node> filterElementNodes(NodeList list) {
        List<Node> result = new ArrayList<Node>(list.getLength());
        for (int i = 0; i < list.getLength(); i++) {
            if (list.item(i).getNodeType() == Node.ELEMENT_NODE) {
                result.add(list.item(i));
            }
        }
        return result;
    }

    /**
     * Returns a node's text content.
     * This method will return the text value represented by the node's direct children.
     * If the given node is a TEXT or CDATA node, then its value is returned.
     *
     * @param n The node.
     * @return The node's text content.
     */
    private static String getNodeTextContents(Node n) {
        if (n.getNodeType() == Node.TEXT_NODE || n.getNodeType() == Node.CDATA_SECTION_NODE) {
            Text txtNode = (Text) n;
            String content = txtNode.getWholeText(); //This concatenates any adjacent text/cdata/entity nodes
            if (content == null)
                return "";
            else
                return content;
        } else {
            if (n.hasChildNodes()) {
                NodeList children = n.getChildNodes();

                for (int i = 0; i < children.getLength(); i++) {
                    //Skip any non-text nodes, like comments or entities
                    Node child = children.item(i);
                    if (child.getNodeType() == Node.TEXT_NODE || child.getNodeType() == Node.CDATA_SECTION_NODE) {
                        Text txtNode = (Text) child;
                        String content = txtNode.getWholeText(); //This concatenates any adjacent text/cdata/entity nodes
                        if (content == null)
                            return "";
                        else
                            return content;
                    }
                }

                return "";
            } else {
                return "";
            }
        }
    }
}
