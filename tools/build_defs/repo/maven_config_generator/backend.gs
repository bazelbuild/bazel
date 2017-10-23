/**
 * @license
 * Copyright 2016 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @fileoverview Backend RPC functions and HTML serving.
 * @author jart@google.com (Justine Tunney)
 */


/**
 * Returns index web page for website.
 * @return {!HtmlService.HtmlOutput}
 */
function doGet() {
  return (HtmlService.createHtmlOutputFromFile('index')
          .setTitle('Bazel Maven Config Generator')
          .setFaviconUrl('https://i.imgur.com/Ute4LEE.png'));
}


/**
 * Returns email address of logged in Google account.
 * @return {string}
 */
function getUserEmail() {
  return Session.getActiveUser().getEmail();
}


/**
 * Fetches {@code url} as text/plain.
 * @param {string} url
 * @return {string}
 */
function fetchTextFromUrl(url) {
  return UrlFetchApp.fetch(url).getAs('text/plain').getDataAsString();
}


/**
 * Fetches {@code url} as binary and slowly calculates hex SHA256 checksum.
 * @param {string} url
 * @return {string}
 */
function getSha256ForUrl(url) {
  var props = PropertiesService.getScriptProperties();
  var key = 'sha256 ' + url;
  var cached = props.getProperty(key);
  if (cached !== null && cached.length == 64) {
    return cached;
  }
  var result = Sha256.hash(UrlFetchApp.fetch(url).getContentText('ISO-8859-1'));
  props.setProperty(key, result);
  return result;
}


/**
 * Stores {@code url} to Google Drive in bazel-mirror/ folder with public read
 * permissions and returns direct URL to file.
 * @param {string} url
 * @return {string}
 */
function mirrorUrl(url) {
  // Note: Drive allows filenames within a folder to contain slashes.
  // Note: Drive allows multiple files within a folder with the same name.
  var lock = LockService.getUserLock();
  var path = 'bazel-mirror/' + url.replace(new RegExp('^https?://'), '');
  var p = path.lastIndexOf('/');
  var dir = path.substr(0, p);
  var folder = DriveApp.getRootFolder();
  var labels = dir.split('/');
  for (var i = 0; i < labels.length; i++) {
    folder = getFolder_(lock, folder, labels[i]);
  }
  var name = path.substr(p + 1);
  var files = folder.getFilesByName(name);
  var file = null;
  while (files.hasNext()) {
    file = files.next();
    if (file.getSize() == 0) {
      file = null;
    }
  }
  // we don't need to lock this, assuming user isn't using multiple tabs
  if (file == null) {
    file = folder.createFile(UrlFetchApp.fetch(url).getBlob());
    file.setSharing(DriveApp.Access.ANYONE_WITH_LINK, DriveApp.Permission.VIEW);
  }
  return 'https://drive.google.com/uc?export=download&id=' + file.getId();
}


/**
 * Returns {@code true} if HEAD request to {@code url} returns 200.
 * @param {string} url
 * @param {boolean}
 */
function doesUrlExist(url) {
  var props = PropertiesService.getScriptProperties();
  var key = 'exists ' + url;
  var cached = props.getProperty(key);
  if (cached !== null) {
    return cached == 'true';
  }
  var status = UrlFetchApp.fetch(url, {'muteHttpExceptions': true}).getResponseCode();
  var result = status == 200;
  if (status == 200 || status == 404) {
    props.setProperty(key, result.toString());
  }
  return result;
}


/**
 * Returns subfolder, creating it if it doesn't exist.
 * @param {!LockService.Lock} lock
 * @param {!DriveApp.Folder} parent
 * @param {string} name
 * @return {!DriveApp.Folder}
 */
function getFolder_(lock, parent, name) {
  if (name == '') {
    return parent;
  }
  var folder = getLiteralFolder_(parent, name);
  if (folder != null) {
    return folder;
  }
  lock.waitLock(10000);
  try {
    folder = getLiteralFolder_(parent, name);
    if (folder != null) {
      return folder;
    }
    return parent.createFolder(name);
  } finally {
    lock.releaseLock();
  }
}


/**
 * Returns subfolder or {@code null} if it doesn't exist.
 * @param {!DriveApp.Folder} parent
 * @param {string} name
 * @return {?DriveApp.Folder}
 */
function getLiteralFolder_(parent, name) {
  var folders = parent.getFoldersByName(name);
  return folders.hasNext() ? folders.next() : null;
}

